#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "index/drip/drip.hpp"

namespace saqlib::drip {
class DripParamBenchmarker {
    // Layer 0 stores ef_base; higher layers store ef_cent,l.
    struct SearchParameterResult {
        std::array<size_t, kMaxLevel> ef_search;
        std::array<size_t, kMaxLevel> nprobe;
        float avg_recall;
        float avg_latency_ms;
        float qps;
        float final_time_s;
        float visited_nodes;
        float dist_cmp_ivf;
        float dist_cmp_graph;
        size_t num_queries;
    };

  public:
    DripParamBenchmarker(FloatRowMat query, UintRowMat gt, int top_k, DRIP *drip_index, int num_threads)
        : query_(std::move(query)), gt_(std::move(gt)), num_threads_(num_threads), top_k_(top_k), drip_index_(drip_index) {
        num_query_ = query_.rows();
    }

    // Scale the paper's ef_base/ef_cent,l and nprobe,l parameters together.
    SearchParameterResult testSearchParameters(double params_k) {
        CHECK_EQ(drip_index_->refinement_budget_, 0) << "HNSW index must be in noconstruct mode for benchmarking.";
        auto original_ef_search = drip_index_->ef_search_;
        auto original_nprobe = drip_index_->nprobe_;
        for (auto &v : drip_index_->ef_search_) {
            v = std::max<size_t>(1, v * params_k);
        }
        for (auto &v : drip_index_->nprobe_) {
            v = v * params_k;
        }

        // Run search_batch with noconstruct=true
        auto tm_start = std::chrono::high_resolution_clock::now();
        auto search_results = drip_index_->search_batch(query_, num_query_, top_k_, num_threads_);
        auto tm_end = std::chrono::high_resolution_clock::now();
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(tm_end - tm_start).count();

        std::swap(drip_index_->ef_search_, original_ef_search);
        std::swap(drip_index_->nprobe_, original_nprobe);

        // Calculate recall and collect metrics
        size_t total_visited_nodes = 0;
        size_t total_dist_cmp_ivf = 0;
        size_t total_dist_cmp_graph = 0;
        // size_t total_probed_clusters = 0;

        std::vector<std::vector<std::pair<float, PID>>> search_results_vec;
        // std::vector<std::vector<std::pair<float, PID>>> graph_candidate_results;
        for (const auto &sr : search_results) {
            std::vector<std::pair<float, PID>> query_result = sr.results;
            query_result.resize(top_k_, std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<PID>::max()));
            LOG_IF(WARNING, query_result.size() < top_k_) << fmt::format("Warning: query result size {} less than top_k {}", query_result.size(), top_k_);
            CHECK_EQ(query_result.size(), top_k_);
            search_results_vec.push_back(query_result);

            for (int i = 0; i <= drip_index_->get_max_level(); ++i) {
                total_visited_nodes += sr.sm.visited_nodes[i];
                total_dist_cmp_ivf += sr.sm.dist_cmp_ivf[i];
                total_dist_cmp_graph += sr.sm.dist_cmp_graph[i];
                // total_probed_clusters += sr.sm.probed_clusters[i];
            }
        }

        float avg_recall = calculateRecall(gt_, search_results_vec);
        // float avg_graph_candidate_recall = calculateRecall(graph_candidate_gt_, graph_candidate_results);
        float avg_latency_ms = 1.0f * elapsed_ms / num_query_;

        SearchParameterResult result;
        result.ef_search = original_ef_search;
        result.nprobe = original_nprobe;
        // result.search_cost_control = FLAGS_search_cost_control;
        result.avg_recall = avg_recall;
        // result.avg_graph_candidate_recall = avg_graph_candidate_recall;
        result.avg_latency_ms = avg_latency_ms;
        result.qps = num_query_ / (elapsed_ms / 1000.0f);
        result.final_time_s = elapsed_ms / 1000.0f;
        result.visited_nodes = 1.0f * total_visited_nodes / num_query_;
        result.dist_cmp_ivf = 1.0f * total_dist_cmp_ivf / num_query_;
        result.dist_cmp_graph = 1.0f * total_dist_cmp_graph / num_query_;
        // result.probed_clusters = 1.0f * total_probed_clusters / num_query_;
        result.num_queries = num_query_;

        LOG(INFO) << fmt::format(
            " ef_search=[{}], nprobe=[{}] := recall={:.4f}, time={:.3f}s, latency={:.2f}ms, visited_nodes={}, dist_cmp_ivf={:.2f}, dist_cmp_graph={:.2f}",
            fmt::join(&result.ef_search[0], &result.ef_search[3], ","), fmt::join(&result.nprobe[0], &result.nprobe[3], ","),
            result.avg_recall, result.final_time_s, result.avg_latency_ms, result.visited_nodes, result.dist_cmp_ivf, result.dist_cmp_graph);

        all_results_.push_back(result);
        return result;
    }

    void benchmarkSearchParameters(float min_recall = 0.85, float max_recall = 0.995, float max_params_k = 10.0, float step = 1.2) {
        CHECK_GT(step, 1.0);
        SearchParameterResult last_result;

        // go min
        last_result.avg_recall = 1.0f;
        for (double params_k = 1.0; last_result.avg_recall > min_recall && params_k > 0; params_k /= step) {
            last_result = testSearchParameters(params_k);
        }

        // go max
        for (double params_k = step; last_result.avg_recall < max_recall && params_k <= max_params_k; params_k *= step) {
            last_result = testSearchParameters(params_k);
        }
    }

    void benchmarkEfsParameters(float min_recall = 0.85, float max_recall = 0.995, float max_params_k = 10.0, float step = 1.2) {
        benchmarkSearchParameters(min_recall, max_recall, max_params_k, step);
    }

    // Write results to CSV
    void writeResultsToCSV(const std::string &filename) {
        std::ofstream csv_file(filename);
        if (!csv_file.is_open()) {
            LOG(ERROR) << "Failed to open CSV file: " << filename << " - " << std::strerror(errno);
            return;
        }

        // Header
        csv_file << "ef_search_by_layer,nprobe_by_layer,avg_recall,avg_latency_ms,qps,final_time_s,visited_nodes,"
                    "dist_cmp_ivf,dist_cmp_graph,num_queries"
                 << std::endl;

        // Data rows
        for (const auto &result : all_results_) {
            csv_file << fmt::format("\"{}\",\"[{}]\",", result.ef_search,
                                    fmt::join(result.nprobe, ","))
                     //  << result.search_cost_control << ","
                     << std::setprecision(6) << result.avg_recall << ","
                     << result.avg_latency_ms << "," << result.qps << ","
                     << result.final_time_s << "," << result.visited_nodes << ","
                     << result.dist_cmp_ivf << "," << result.dist_cmp_graph << ","
                     << result.num_queries << std::endl;
        }

        csv_file.close();
        LOG(INFO) << fmt::format("Results written to: {}", filename);
    }

  private:
    static float calculateRecall(const UintRowMat &gt, const std::vector<std::vector<std::pair<float, PID>>> &search_results) {
        if (gt.rows() == 0 || search_results.empty()) {
            return 0.0f;
        }
        size_t total_recall = 0;
        size_t total_expected = 0;

        for (size_t query_idx = 0; query_idx < search_results.size() && query_idx < gt.rows(); ++query_idx) {
            const auto &query_results = search_results[query_idx];
            const auto gt_row = gt.row(query_idx);

            size_t found_count = 0;
            for (long gt_idx = 0; gt_idx < gt_row.size() && gt_idx < query_results.size(); ++gt_idx) {
                PID gt_id = gt_row(gt_idx);
                for (const auto &result : query_results) {
                    if (result.second == gt_id) {
                        found_count++;
                        break;
                    }
                }
            }

            total_recall += found_count;
            total_expected += std::min((size_t)query_results.size(), (size_t)gt_row.size());
        }

        return total_expected > 0 ? (float)total_recall / total_expected : 0.0f;
    }

    FloatRowMat query_;
    size_t num_query_;
    UintRowMat gt_;
    int num_threads_, top_k_;
    DRIP *drip_index_;

    std::vector<SearchParameterResult> all_results_;
};
} // namespace saqlib::drip
