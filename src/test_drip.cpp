#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "data_options.h"

#include "defines.hpp"
#include "drip_param_benchmarker.hpp"
#include "index/drip/drip.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"
#include "utils/tools.hpp"

DEFINE_int32(M, 64, "HNSW graph degree");
DEFINE_int32(ef_build, 64, "HNSW construction beam width (paper: ef_build)");
DEFINE_int32(ef_assign, 10, "Cluster-assignment beam width (paper: ef_assign)");
DEFINE_int32(init, 2, "Initial hybrid layer (paper: init)");

DEFINE_int32(target_numq, 1'000'000, "");
DEFINE_int32(ef_base, 400, "Base-layer search beam width (paper: ef_base)");
DEFINE_double(p_probe, 0.06, "Initial cluster probing ratio (paper: p_probe)");
DEFINE_double(qps_test_qsr, 1.0, "query sample rate for final qps-recall test. 0 means disable qps test.");
DEFINE_int32(top_k, 100, "");

DEFINE_int32(theta1, 800, "Layer-1 cluster splitting threshold (paper: Theta_1)");
DEFINE_int32(budget, 1200, "Per-query refinement budget (paper: budget)");
DEFINE_double(early_drip_tau, 0.5, "Early-dripping ratio tau in theta_l = tau * Theta_l");
DEFINE_int32(leisure_cracking_after, 1'000, "");
DEFINE_bool(adaptive_refinement, true, "Use cluster visit counts to select refinement work");
DEFINE_bool(probe_cen, false, "");
DEFINE_uint64(random_seed, 42, "Seed used by DRIP level assignment");

// Deprecated command-line aliases retained for existing experiment scripts.
DEFINE_int32(efc, 64, "Deprecated alias for --ef_build");
DEFINE_int32(efc_ivf, 10, "Deprecated alias for --ef_assign");
DEFINE_int32(init_level, 2, "Deprecated alias for --init");
DEFINE_int32(efs_graph, 400, "Deprecated alias for --ef_base");
DEFINE_double(efs_cpr, 0.06, "Deprecated alias for --p_probe");
DEFINE_int32(drip_threshold, 0, "Deprecated alias for --theta1");
DEFINE_int32(refine_budget, 1200, "Deprecated alias for --budget");

using namespace saqlib::drip;
using namespace saqlib;

template <typename T>
T resolve_legacy_flag(const char *paper_name, T paper_value, const char *legacy_name, T legacy_value) {
    const auto paper_info = GFLAGS_NAMESPACE::GetCommandLineFlagInfoOrDie(paper_name);
    const auto legacy_info = GFLAGS_NAMESPACE::GetCommandLineFlagInfoOrDie(legacy_name);
    if (legacy_info.is_default) {
        return paper_value;
    }
    CHECK(paper_info.is_default || paper_value == legacy_value)
        << "Conflicting --" << paper_name << " and deprecated --" << legacy_name;
    LOG(WARNING) << "--" << legacy_name << " is deprecated; use --" << paper_name;
    return legacy_value;
}

struct DripParameters {
    int ef_build;
    int ef_assign;
    int init;
    int ef_base;
    double p_probe;
    int theta1;
    int budget;
    double tau;
};

DripParameters resolve_drip_parameters() {
    return {
        .ef_build = resolve_legacy_flag("ef_build", FLAGS_ef_build, "efc", FLAGS_efc),
        .ef_assign = resolve_legacy_flag("ef_assign", FLAGS_ef_assign, "efc_ivf", FLAGS_efc_ivf),
        .init = resolve_legacy_flag("init", FLAGS_init, "init_level", FLAGS_init_level),
        .ef_base = resolve_legacy_flag("ef_base", FLAGS_ef_base, "efs_graph", FLAGS_efs_graph),
        .p_probe = resolve_legacy_flag("p_probe", FLAGS_p_probe, "efs_cpr", FLAGS_efs_cpr),
        .theta1 = resolve_legacy_flag("theta1", FLAGS_theta1, "drip_threshold", FLAGS_drip_threshold),
        .budget = resolve_legacy_flag("budget", FLAGS_budget, "refine_budget", FLAGS_refine_budget),
        .tau = FLAGS_early_drip_tau,
    };
}

class HNSWCrackingTester {
  public:
    explicit HNSWCrackingTester(DripParameters parameters)
        : parameters_(parameters) {}

    // Real data specific parameters
    size_t num_dim_;
    size_t num_data_;
    size_t num_query_;

    // HNSW parameters for real data
    int top_k_ = FLAGS_top_k;
    size_t random_seed_ = FLAGS_random_seed;

    // Common HNSW components
    std::unique_ptr<DRIP> drip_index_;
    DripParameters parameters_;

    struct QueryMetrics {
        float recall;
        float latency_ms;
        double accumulate_time_ms;
        SearchMetrics sm;
        IndexMetrics im;
    };

    FloatRowMat data_;
    FloatRowMat query_;
    UintRowMat gt_;
    std::vector<std::vector<PID>> filtered_gt_;
    FloatRowMat data_vars_;

    // Helper function to load test data
    void loadTestData() {
        DataFilePaths paths;
        if (*paths.data_file.rbegin() == '.') {
            utils::load_something_series<float, FloatRowMat>(paths.data_file.c_str(), data_);
        } else {
            utils::load_something<float, FloatRowMat>(paths.data_file.c_str(), data_);
        }
        utils::load_something<float, FloatRowMat>(paths.query_file.c_str(), query_);

        if (utils::file_exists(paths.gt_file.c_str())) {
            utils::load_something<PID, UintRowMat>(paths.gt_file.c_str(), gt_);
        } else {
            LOG(WARNING) << fmt::format("Ground truth file {} not found, skipping.", paths.gt_file);
        }

        num_dim_ = data_.cols();
        num_data_ = data_.rows();

        // Adjust GT
        {
            auto gt_c = std::move(gt_);
            gt_ = gt_c.leftCols(top_k_);
        }

        // auto repeat_query = [&](int f) {
        //     auto qc = std::move(query_);
        //     query_ = qc.replicate(f, 1);
        //     auto gt_c = std::move(gt_);
        //     gt_ = gt_c.replicate(f, 1);
        // };
        // if (FLAGS_query_sample_rate <= 1) {
        //     num_query_ = query_.rows() * FLAGS_query_sample_rate;
        //     auto qc = std::move(query_);
        //     query_ = qc.topRows(num_query_);
        //     auto gt_c = std::move(gt_);
        //     gt_ = gt_c.topRows(num_query_);
        // } else {
        //     LOG(INFO) << "FLAGS_query_sample_rate > 1，replicate queries";
        //     num_query_ = query_.rows() * FLAGS_query_sample_rate;

        //     int f = static_cast<int>(std::ceil(FLAGS_query_sample_rate));
        //     repeat_query(f);
        //     CHECK_GE(query_.rows(), num_query_);
        // }

        num_query_ = FLAGS_target_numq;
        // if (FLAGS_target_numq > num_query_) {
        //     int f = static_cast<int>(std::ceil(static_cast<double>(FLAGS_target_numq) / num_query_));
        //     LOG(INFO) << "FLAGS_target_numq > num_query_, replicate queries by factor " << f;
        //     repeat_query(f);
        //     num_query_ = FLAGS_target_numq;
        // }

        LOG(INFO) << "query shape: " << query_.rows() << "x" << query_.cols();
        LOG(INFO) << "gt shape: " << gt_.rows() << "x" << gt_.cols();
    }

    void filterRealDataGroundTruth(size_t subset_size) {
        LOG(INFO) << "Filtering ground truth to include only subset points...  GT: " << gt_.rows() << "x" << gt_.cols();

        // std::vector<std::vector<PID>> filtered_gt(num_query_ori_);
        filtered_gt_.resize(query_.rows());
        size_t total_filtered_results = 0;

        for (long query_idx = 0; query_idx < query_.rows(); ++query_idx) {
            auto &cur_gt = filtered_gt_[query_idx];
            for (long gt_idx = 0; gt_idx < gt_.cols() && cur_gt.size() < top_k_; ++gt_idx) {
                PID gt_point_id = gt_(query_idx, gt_idx);
                if (gt_point_id < subset_size) {
                    cur_gt.push_back(gt_point_id);
                }
            }
            total_filtered_results += cur_gt.size();
        }

        LOG(INFO) << fmt::format("Ground truth filtering completed. Total valid results: {} (avg {:.1f} per query)",
                                 total_filtered_results, static_cast<float>(total_filtered_results) / num_query_);
    }

    auto calculateRealDataRecall(const std::vector<std::pair<float, PID>> &query_results,
                                 const std::vector<PID> &cur_gt) {
        size_t found_count = 0;
        for (size_t i = 0; i < cur_gt.size(); ++i) {
            PID gt_id = cur_gt[i];
            bool found = false;
            for (const auto &result : query_results) {
                if (result.second == gt_id) {
                    found = true;
                    break;
                }
            }
            if (found) {
                found_count++;
            }
        }
        return found_count;
    }

    float calculateRealDataRecall(const std::vector<std::vector<std::pair<float, PID>>> &search_results,
                                  const std::vector<std::vector<PID>> &filtered_gt) {
        size_t total_recall = 0;
        size_t total_expected = 0;

        for (size_t query_idx = 0; query_idx < search_results.size(); ++query_idx) {
            const auto &query_results = search_results[query_idx];
            const auto &cur_gt = filtered_gt[query_idx];

            size_t found_count = calculateRealDataRecall(query_results, cur_gt);

            total_recall += found_count;
            total_expected += cur_gt.size();
        }

        return total_expected > 0 ? static_cast<float>(total_recall) / total_expected : 0.0f;
    }

    float build_drip_index() {
        filterRealDataGroundTruth(num_data_);

        // Create HNSW index
        drip_index_ = std::make_unique<DRIP>(
            num_dim_, num_data_ + 10, FLAGS_M, parameters_.ef_build, parameters_.ef_assign, random_seed_, DistType::L2Sqr);

        // Construct the index
        LOG(INFO) << fmt::format("Building HNSW index with {} data points...", num_data_);
        auto tm_start = std::chrono::high_resolution_clock::now();
        drip_index_->construct(num_data_, data_, parameters_.init, FLAGS_num_threads);
        auto tm_end = std::chrono::high_resolution_clock::now();
        auto tm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();
        LOG(INFO) << fmt::format("HNSW index construction completed. Time taken: {}s",
                                 tm_ms / 1000.0);
        auto im = drip_index_->get_index_metrics();
        LOG(INFO) << "unconstruct_points_cnt_: " << im.clus_points_cnt;
        LOG(INFO) << "construct_points_cnt_: " << im.graph_points_cnt;

        return tm_ms;
    }

    void set_build_args() {
        const int ori_preb_level = drip_index_->get_refinement_frontier();
        if (FLAGS_probe_cen)
            drip_index_->refinement_frontier_ = 2;
        drip_index_->set_search_parameters(parameters_.ef_base, parameters_.p_probe);
        if (FLAGS_probe_cen)
            drip_index_->refinement_frontier_ = ori_preb_level;

        LOG(INFO) << "ef_base / ef_cent,l: " << drip_index_->ef_search_[0] << ", " << drip_index_->ef_search_[1] << ", " << drip_index_->ef_search_[2] << ", " << drip_index_->ef_search_[3];
        LOG(INFO) << "nprobe,l: " << drip_index_->nprobe_[0] << ", " << drip_index_->nprobe_[1] << ", " << drip_index_->nprobe_[2] << ", " << drip_index_->nprobe_[3];

        drip_index_->set_refinement_parameters(parameters_.budget, parameters_.theta1, parameters_.tau);
        drip_index_->set_leisure_cracking(FLAGS_leisure_cracking_after);
        drip_index_->set_adaptive_refinement(FLAGS_adaptive_refinement);
        LOG(INFO) << "Adaptive refinement: "
                  << (FLAGS_adaptive_refinement ? "enabled" : "disabled (fixed PID-order leisure refinement)");
        LOG(INFO) << fmt::format(
            "Refinement parameters: budget={}, Theta_1={}, tau={}, theta_1={}",
            parameters_.budget,
            parameters_.theta1,
            parameters_.tau,
            parameters_.tau * parameters_.theta1);
    }

    std::vector<QueryMetrics> runSearch() {
        std::vector<QueryMetrics> query_metrics(num_query_);

        BS::thread_pool thread_pool(FLAGS_num_threads);
        auto query_start_time = std::chrono::high_resolution_clock::now();
        auto futures = thread_pool.submit_loop(
            0,
            num_query_,
            [&](size_t idx) {
                auto tm_start = std::chrono::high_resolution_clock::now();
                auto ret = drip_index_->search_knn_cracking(query_.row(idx % query_.rows()), top_k_);
                auto tm_end = std::chrono::high_resolution_clock::now();
                query_metrics[idx].im = drip_index_->get_index_metrics();
                if (ret.results.size() > top_k_) {
                    ret.results.resize(top_k_);
                }
                query_metrics[idx].sm = ret.sm;
                query_metrics[idx].latency_ms = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - tm_start).count() / 1000.0;
                query_metrics[idx].accumulate_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(tm_end - query_start_time).count() / 1000.0;

                query_metrics[idx].recall = calculateRealDataRecall(ret.results, filtered_gt_[idx % filtered_gt_.size()]) * 1.0 / top_k_;

                float threshold = 0.2f;
                if (query_metrics[idx].recall < threshold) {
                    std::string logstr;
                    for (size_t i = 0; i < ret.results.size() && i < 10; i++) {
                        logstr += fmt::format("{}({:.2f}), ", ret.results[i].second, ret.results[i].first);
                    }
                    logstr += "\n";
                    for (long i = 0; i < filtered_gt_[idx % filtered_gt_.size()].size() && i < top_k_ && i < 10; i++) {
                        logstr += fmt::format("{}, ", filtered_gt_[idx % filtered_gt_.size()][i]);
                    }
                    LOG_FIRST_N(WARNING, 10) << "Query " << idx << " recall is less than " << threshold << ": " << query_metrics[idx].recall << " \n"
                                             << logstr;
                    // auto ret = drip_index_->search_knn_cracking(query_.row(idx % query_.rows()), top_k_);
                }
                if (FLAGS_target_numq > 10 && idx % (FLAGS_target_numq / 10) == 0) {
                    LOG(INFO) << "finished id:" << idx;
                }
            });
        futures.wait();

        std::sort(query_metrics.begin(), query_metrics.end(), [](const QueryMetrics &a, const QueryMetrics &b) {
            return a.accumulate_time_ms < b.accumulate_time_ms;
        });

        float total_latency_ms = 0;
        float recall = 0;
        SearchMetrics tot_metrics;
        for (size_t idx = 0; idx < num_query_; idx++) {
            tot_metrics.accumulate(query_metrics[idx].sm);
            total_latency_ms += query_metrics[idx].latency_ms;
            recall += query_metrics[idx].recall;
            // LOG(INFO) << "Query " << idx << " visited nodes: " << query_metrics[idx].visited_nodes;
        }
        recall /= num_query_;
        float recall_var = 0;
        for (size_t idx = 0; idx < num_query_; idx++) {
            float diff = query_metrics[idx].recall - recall;
            recall_var += diff * diff;
        }
        recall_var /= num_query_;

        float single_thread_qps = num_query_ / total_latency_ms * 1e3;
        float tot_node_vis = 0;
        float tot_dist_cmp_ivf = 0;
        float tot_dist_cmp_graph = 0;
        std::string logstr;
        for (int level = drip_index_->get_max_level(); level >= 0; level--) {
            float visited_nodes = 1.0f * tot_metrics.visited_nodes[level] / num_query_;
            float dist_cmp_ivf = 1.0f * tot_metrics.dist_cmp_ivf[level] / num_query_;
            float dist_cmp_graph = 1.0f * tot_metrics.dist_cmp_graph[level] / num_query_;
            tot_node_vis += visited_nodes;
            tot_dist_cmp_ivf += dist_cmp_ivf;
            tot_dist_cmp_graph += dist_cmp_graph;
            if (level <= drip_index_->refinement_frontier_)
                logstr += fmt::format("[{}] {} /{}, ", level, dist_cmp_graph, dist_cmp_ivf);
        }
        logstr = fmt::format("EF {}-{}: recall {:.3f} | tot_visited_nodes {}, tot_dist_compute graph/ivf: {} / {} | {}",
                             parameters_.ef_base, parameters_.p_probe, recall, tot_node_vis, tot_dist_cmp_graph, tot_dist_cmp_ivf, logstr);

        LOG(INFO) << "=======> " << logstr;
        LOG(INFO) << fmt::format("=======> QPS on full query (EF {}-{}): recall {:.3f} ± {:.4f}, SingleThread_QPS {:.1f}",
                                 parameters_.ef_base, parameters_.p_probe, recall, std::sqrt(recall_var), single_thread_qps);
        // LOG(INFO) << fmt::format(" probe_clusters per query at each level: {},{},{},{}",
        //                          1.0f * tot_metrics.probed_clusters[0] / num_query_,
        //                          1.0f * tot_metrics.probed_clusters[1] / num_query_,
        //                          1.0f * tot_metrics.probed_clusters[2] / num_query_,
        //                          1.0f * tot_metrics.probed_clusters[3] / num_query_);

        auto im = drip_index_->get_index_metrics();
        LOG(INFO) << fmt::format("=======> Index metrics: clus_points_cnt: {}, graph_points_cnt: {}",
                                 im.clus_points_cnt, im.graph_points_cnt);
        return query_metrics;
    }

    void BnechmarkDrip(const std::string &qps_csv_path, float min_recall = 0.85, float max_recall = 0.995, float max_params_k = 50.0, float step = 1.5) {
        drip_index_->refinement_budget_ = 0; // disable cracking
        auto nq = query_.rows() * FLAGS_qps_test_qsr;
        DripParamBenchmarker benchmarker(query_.topRows(nq), gt_.topRows(nq), top_k_, drip_index_.get(), FLAGS_num_threads);
        benchmarker.benchmarkSearchParameters(min_recall, max_recall, max_params_k, step);
        benchmarker.writeResultsToCSV(qps_csv_path);
    }
};

void to_csv(const std::string &csv_path, std::vector<HNSWCrackingTester::QueryMetrics> &query_metrics, float build_tm_ms) {

    std::ofstream csv_file(csv_path);
    CHECK(csv_file.is_open()) << "Failed to open CSV file: " << csv_path << " - " << std::strerror(errno);
    csv_file << std::setprecision(15);
    csv_file << "query_idx,cummulative_time_ms,cummulative_wall_time_ms,latency_ms,recall,visited_nodes,dist_cmp_ivf,dist_cmp_graph,crack_cost,"
             << "idx_clus_points_cnt,idx_graph_points_cnt,idx_construct_progress,idx_graph_size_0,idx_graph_size_1,idx_clus_size_1,idx_clus_size_2"
             << std::endl;
    double tm_sum = static_cast<double>(build_tm_ms) * FLAGS_num_threads;
    LOG(INFO) << "[debug] build_tm_ms: " << build_tm_ms << ", query_metrics[0].latency_ms: " << query_metrics[0].latency_ms << ", tm_sum: " << tm_sum;
    for (size_t i = 0; i < query_metrics.size(); ++i) {
        size_t visited_nodes_tot = 0;
        size_t dist_cmp_ivf_tot = 0;
        size_t dist_cmp_graph_tot = 0;
        for (size_t j = 0; j < 5; j++) {
            visited_nodes_tot += query_metrics[i].sm.visited_nodes[j];
            dist_cmp_ivf_tot += query_metrics[i].sm.dist_cmp_ivf[j];
            dist_cmp_graph_tot += query_metrics[i].sm.dist_cmp_graph[j];
        }
        tm_sum += query_metrics[i].latency_ms;

        csv_file << i << ","
                 << tm_sum << ","
                 << static_cast<double>(build_tm_ms) + query_metrics[i].accumulate_time_ms << ","
                 << query_metrics[i].latency_ms << ","
                 << query_metrics[i].recall << ","
                 << visited_nodes_tot << ","
                 << dist_cmp_ivf_tot << ","
                 << dist_cmp_graph_tot << ","
                 //  << query_metrics[i].sm.probed_clusters[1] << ","
                 //  << query_metrics[i].sm.probed_clusters[2] << ","
                 << query_metrics[i].sm.crack_cost << ","

                 << query_metrics[i].im.clus_points_cnt << ","
                 << query_metrics[i].im.graph_points_cnt << ","
                 << query_metrics[i].im.construct_progress << ","
                 << query_metrics[i].im.graph_size[0] << ","
                 << query_metrics[i].im.graph_size[1] << ","
                 << query_metrics[i].im.cluster_size[1] << ","
                 << query_metrics[i].im.cluster_size[2];

        csv_file << std::endl;
    }
    csv_file.close();
    std::cout << "Query metrics CSV saved to: " << csv_path << std::endl;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    const auto parameters = resolve_drip_parameters();
    HNSWCrackingTester tester(parameters);

    tester.loadTestData();
    auto build_tm_ms = tester.build_drip_index();
    LOG(INFO) << "Build time: " << build_tm_ms << "ms";
    tester.set_build_args();

    auto query_metrics = tester.runSearch();
    LOG(INFO) << "Search finished.";

    DataFilePaths paths("drip");
    // results file path
    std::string csv_path = fmt::format("{}/{}{}", paths.result_path, FLAGS_dataset, FLAGS_query_type.empty() ? "" : ("_" + FLAGS_query_type));

    csv_path += fmt::format("_M{}", FLAGS_M);
    csv_path += fmt::format("_efbuild{}_efassign{}", parameters.ef_build, parameters.ef_assign);
    csv_path += fmt::format("_topk{}", FLAGS_top_k);
    csv_path += fmt::format("_efbase{}_pprobe{}", parameters.ef_base, parameters.p_probe);
    csv_path += fmt::format("_init{}", parameters.init);
    csv_path += fmt::format("_theta1{}", parameters.theta1);
    csv_path += fmt::format("_budget{}", parameters.budget);
    csv_path += fmt::format("_tau{}", parameters.tau);
    csv_path += fmt::format("_leisurec{}", FLAGS_leisure_cracking_after);
    if (!FLAGS_adaptive_refinement) {
        csv_path += "_noadaptive";
    }
    if (FLAGS_probe_cen) {
        csv_path += "_probecen";
    }
    std::string curr_timestamp = utils::current_datetime_str();
    csv_path += fmt::format("_th{}.v11.{}.csv", FLAGS_num_threads, curr_timestamp);

    // Output query_metrics as CSV
    LOG(INFO) << "Saving query metrics to CSV: " << csv_path;
    to_csv(csv_path, query_metrics, build_tm_ms);

    if (FLAGS_qps_test_qsr > 0) {
        auto qps_csv_path = csv_path;
        qps_csv_path.replace(qps_csv_path.end() - 4, qps_csv_path.end(), ".qps.csv");
        tester.BnechmarkDrip(qps_csv_path, 0.75, 0.995);
    }

    google::ShutdownGoogleLogging();
    return 0;
}
