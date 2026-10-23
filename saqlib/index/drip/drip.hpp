// HNSW is developed from the [HNSW library](https://github.com/nmslib/hnswlib)
#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/StopW.hpp"
#include "utils/atomic_counter.hpp"
#include "utils/buffer.hpp"
#include "utils/concurrent_vector.hpp"
#include "utils/tools.hpp"
#include "utils/visited_pool.hpp"

#include "hnsw.hpp"

namespace saqlib::drip {

struct IndexMetricsAtomic {
    std::array<utils::AtomicCounter, kMaxLevel + 1> graph_size;
    std::array<utils::AtomicCounter, kMaxLevel + 1> cluster_size;
};
struct IndexMetrics {
    size_t clus_points_cnt{0};
    size_t graph_points_cnt{0};
    // size_t construct_computation_cnt{0};
    float construct_progress{0};
    std::array<size_t, kMaxLevel + 1> graph_size{0};
    std::array<size_t, kMaxLevel + 1> cluster_size{0};
};

struct SearchResult {
    std::vector<std::pair<float, PID>> results;
    buffer::SearchBufferBig graph_candidates;
    SearchMetrics sm;
};

class DRIP : public HNSW {
  public:
    explicit DRIP() = default;
    explicit DRIP(size_t num_dim, size_t max_elements, size_t M, size_t ef_build, size_t ef_assign,
                  size_t random_seed = 100, DistType metric_type = DistType::L2Sqr)
        : HNSW(num_dim, max_elements, M, ef_build, random_seed, metric_type),
          hang_at_ep(max_elements) {
        ef_assign_ = ef_assign;

        // search parameters
        nprobe_.fill(0);
        ef_search_.fill(1);

        // refinement parameters
        for (size_t i = 0; i < cluster_at_.size(); i++) {
            cluster_at_[i].resize(max_elements);
        }
        splitting_threshold_.fill(kDefaultSplittingThreshold);
    }
    ~DRIP() = default;

    IndexMetrics get_index_metrics() const {
        IndexMetrics im{};
        size_t traverse_cnt = 0;
        for (int l = 0; l <= maxlevel_; l++) {
            im.graph_size[l] = index_metrics_.graph_size[l].load();
            im.cluster_size[l] = index_metrics_.cluster_size[l].load();
            im.clus_points_cnt += im.cluster_size[l];
            im.graph_points_cnt += im.graph_size[l];
            traverse_cnt += im.cluster_size[l] * (maxlevel_ - l);
        }
        traverse_cnt += im.graph_points_cnt * (maxlevel_ + 1);
        im.construct_progress = static_cast<float>(traverse_cnt) / (max_elements_ * (maxlevel_ + 1.0));
        return im;
    }
    int get_refinement_frontier() const {
        return refinement_frontier_.load(std::memory_order_acquire);
    }
    int get_prebuild_level() const {
        return get_refinement_frontier();
    }

    void set_search_parameters(int ef_base, float p_probe = 0.06, float beta = 1.5, float gamma = 0.61) {
        const int refinement_level = refinement_frontier_.load(std::memory_order_relaxed);
        nprobe_[refinement_level] = static_cast<size_t>((p_probe * max_elements_) / std::pow(M_, refinement_level) + 1);
        ef_search_[refinement_level] = static_cast<size_t>(static_cast<float>(nprobe_[refinement_level]) * beta + 1);
        for (auto l = refinement_level - 1; l > 0; l--) {
            nprobe_[l] = static_cast<size_t>(static_cast<float>(nprobe_[l + 1]) * std::pow(M_, gamma) + 1);
            ef_search_[l] = static_cast<size_t>(static_cast<float>(nprobe_[l]) * beta + 1);
        }
        ef_search_[0] = ef_base;
    }
    void set_search_parameters(std::array<size_t, kMaxLevel> ef_search, std::array<size_t, kMaxLevel> nprobe) {
        ef_search_ = std::move(ef_search);
        nprobe_ = std::move(nprobe);
    }
    void set_refinement_parameters(int budget, int theta1, float early_drip_tau = 0.5) {
        CHECK_GE(budget, 0);
        CHECK_GE(theta1, 0);
        CHECK_GE(early_drip_tau, 0.0);
        CHECK_LE(early_drip_tau, 1.0);
        refinement_budget_ = budget;
        early_drip_tau_ = early_drip_tau;
        splitting_threshold_[1] = theta1;
        for (int l = 2; l < refinement_frontier_ + 1; l++) {
            splitting_threshold_[l] = static_cast<int>(std::max<size_t>(1, splitting_threshold_[l - 1] / M_));
        }
    }
    void set_refine_parameters(int budget, int theta1, float early_drip_tau = 0.5) {
        set_refinement_parameters(budget, theta1, early_drip_tau);
    }
    void set_leisure_cracking(int64_t after_nquery) {
        leisure_cracking_after_ = after_nquery;
    }
    void set_adaptive_refinement(bool enabled) {
        adaptive_refinement_ = enabled;
    }

    void construct(size_t data_num, const FloatRowMat &data, int init, size_t num_threads = 0);

    SearchResult search_knn_cracking(const FloatVec &query, size_t topk_sz);

    std::vector<SearchResult> search_batch(const FloatRowMat &query, size_t query_num, size_t topk_sz, int num_threads) {
        std::vector<SearchResult> results(query_num);

        BS::thread_pool thread_pool(num_threads);
        auto futures = thread_pool.submit_loop(
            0,
            query_num,
            [&](size_t idx) {
                results[idx] = search_knn_cracking(query.row(idx), topk_sz);
            });
        futures.wait();
        return results;
    }

  public:
    static constexpr size_t kDefaultSplittingThreshold = 400;
    static constexpr int kMaxUnbuildLevel = 3;
    static constexpr int kGraphInsertionCostRho = 10; // Paper constant rho: graph insertion vs. reassignment work.

    struct Cluster {
        utils::AtomicCounter visited_cnt{0};
        utils::ConcurrentVector<ToBeAddPoint> n1_points;
        utils::ConcurrentVector<ToBeAddPoint> n2_points;

        void add_point(const ToBeAddPoint &p) {
            if (p.c_level == p.p_level + 1) {
                n1_points.push_back(p);
            } else {
                n2_points.push_back(p);
            }
        }

        size_t size() const {
            return n1_points.size() + n2_points.size();
        }
    };

    mutable IndexMetricsAtomic index_metrics_;

    // ===== construction parameters =====
    size_t ef_assign_{10};

    // ===== progressive-refinement status =====
    std::atomic<int> refinement_frontier_{2}; // Paper symbol f; maxlevel_ is the fixed highest layer H.
    int initial_refinement_frontier_ = 2;
    mutable std::mutex refinement_frontier_mutex_;
    std::array<std::vector<Cluster>, kMaxUnbuildLevel + 1> cluster_at_;
    std::vector<std::atomic<uint64_t>> hang_at_ep;
    std::atomic<size_t> leisure_position_{0};
    std::atomic<size_t> served_query_{0};

    // ===== query parameters =====
    std::array<size_t, kMaxLevel> nprobe_;  // n_probe,l; index 0 is unused.
    std::array<size_t, kMaxLevel> ef_search_; // index 0: ef_base; index l > 0: ef_cent,l.

    // ===== refinement parameters =====
    int64_t leisure_cracking_after_ = 1'000;
    size_t refinement_budget_ = 1200;
    float early_drip_tau_ = 0.5;
    bool adaptive_refinement_ = true;
    std::array<size_t, kMaxUnbuildLevel + 1> splitting_threshold_;

  protected:
    std::optional<ToBeAddPoint> add_new_vector(PID cur_point, const FloatVec &vec);

    void add_node_into_clus(int level, PID curr_ep, const ToBeAddPoint &p) {
        DCHECK_GT(level, 0);
        auto &clus = cluster_at_[level][curr_ep];
        clus.add_point(p);
        hang_at_ep[p.point] = utils::pack_2_uint32(level, curr_ep);
    }

    size_t construct_nextlevel_graph_from_clus(PID cur_ivf, int level, size_t con_sz = -1) {
        size_t construct_cnt = 0;
        if (level < 1 || con_sz < 1)
            return construct_cnt;
        auto next_level = level - 1;
        auto &clus = cluster_at_[level][cur_ivf];
        auto &n1p = clus.n1_points;
        if (n1p.empty())
            return construct_cnt;
        while (auto p_opt = n1p.pop_back()) {
            auto p = p_opt.value();
            CHECK(p.point && p.p_level == next_level) << "Point " << p.point << " level mismatch: " << p.p_level << " and " << next_level;

            if (uint64_t packed_value = utils::pack_2_uint32(p.c_level, p.ep);
                !hang_at_ep[p.point].compare_exchange_strong(packed_value, 0)) {
                if (packed_value == 0)
                    continue;
                auto [curr_level, curr_ep] = utils::unpack_2_uint32(hang_at_ep[p.point]);
                LOG(ERROR) << "Point " << p.point << fmt::format("; hang_at_ep do not match. should be {}/{}, but is {}/{}.", p.ep, p.c_level, curr_ep, curr_level);
            }

            p.c_level--;
            CHECK_EQ(p.p_level, p.c_level) << "Point " << p.point << " level mismatch: " << pid2level_[p.point] << " and " << p.c_level;
            // LOG(INFO) << "Level / point : " << p.level << " / " << p.point << " mutually_connect_new_element";

            construct_node_2g(p.point, p.c_level, p.ep);

            ++construct_cnt;
            if (con_sz <= construct_cnt)
                break;
        }

        index_metrics_.cluster_size[level] -= construct_cnt;
        index_metrics_.graph_size[level - 1] += construct_cnt;
        return construct_cnt;
    }

    size_t construct_nextlevel_clus_from_clus(PID cur_ivf, int level, size_t con_sz = -1) {
        size_t construct_cnt = 0;
        auto &n2p = cluster_at_[level][cur_ivf].n2_points;
        if (n2p.empty() || con_sz < 1)
            return construct_cnt;
        auto next_level = level - 1;
        buffer::SearchBufferBig candidate_set(ef_assign_);
        SearchMetrics metrics;
        while (auto p_opt = n2p.pop_back()) {
            auto p = p_opt.value();
            CHECK_LT(p.p_level, next_level) << "next_level: " << next_level << " p.level: " << pid2level_[p.point];
            CHECK(p.point) << "Point " << p.point << " is not valid";
            if (uint64_t packed_value = utils::pack_2_uint32(p.c_level, p.ep);
                !hang_at_ep[p.point].compare_exchange_strong(packed_value, 0)) {
                if (packed_value == 0)
                    continue;
                auto [curr_level, curr_ep] = utils::unpack_2_uint32(hang_at_ep[p.point]);
                LOG(WARNING) << "Point " << p.point << fmt::format(" at {}/{}; hang_at_ep do not match the value. should be {}/{}, but is {}/{}.", cur_ivf, level, p.c_level, p.ep, curr_level, curr_ep);
            }
            candidate_set.clear();
            candidate_set.insert(p.ep, p.dist);
            auto curr_vec = get_l0_vec_data(p.point);
            // find and hang under the nearest entry point
            traverse_at_level(curr_vec, next_level, candidate_set, nullptr, metrics);

            auto lowest_entry = candidate_set.lowest_entry();
            auto curr_ep = lowest_entry.id;
            auto curr_dist = lowest_entry.distance;
            p.ep = curr_ep;
            p.c_level = next_level;
            p.dist = curr_dist;
            CHECK_NE(curr_ep, p.point) << "hang at it self: " << p.point << " -> " << curr_ep;

            add_node_into_clus(next_level, curr_ep, p);

            construct_cnt++;
            if (con_sz <= construct_cnt)
                break;
        }
        index_metrics_.cluster_size[level] -= construct_cnt;
        index_metrics_.cluster_size[next_level] += construct_cnt;
        return construct_cnt;
    }

    void leisure_dripping(int level, int64_t quota, SearchMetrics &metrics, bool strict_budget = false) {
        if (quota < 1)
            return;
        size_t consecutive_empty_clusters = 0;
        for (; quota > 0 && index_metrics_.cluster_size[level];) {
            auto position = leisure_position_.load();
            auto pid = position % max_elements_;
            bool did_work = false;
            if (auto sz = cluster_at_[level][pid].n1_points.size(); sz) {
                const auto graph_quota = strict_budget
                                             ? static_cast<size_t>(quota) / kGraphInsertionCostRho
                                             : static_cast<size_t>(quota);
                if (!graph_quota)
                    break;
                sz = construct_nextlevel_graph_from_clus(pid, level, graph_quota);
                quota -= static_cast<int64_t>(sz * kGraphInsertionCostRho);
                metrics.crack_cost += sz * kGraphInsertionCostRho;
                did_work = sz > 0;

                if (strict_budget && !cluster_at_[level][pid].n1_points.empty()) {
                    if (quota < kGraphInsertionCostRho)
                        break;
                    continue;
                }
            }
            if (auto sz = cluster_at_[level][pid].n2_points.size(); quota && sz) {
                sz = construct_nextlevel_clus_from_clus(pid, level, quota);
                quota -= static_cast<int64_t>(sz);
                metrics.crack_cost += sz;
                did_work = did_work || sz > 0;
            }
            if (quota <= 0)
                break;
            if (leisure_position_.compare_exchange_strong(position, position + 1)) {
                consecutive_empty_clusters = did_work ? 0 : consecutive_empty_clusters + 1;
                if (consecutive_empty_clusters >= max_elements_) {
                    LOG(WARNING) << "Leisure refinement stopped after scanning all clusters without progress. "
                                 << fmt::format("dim: {} quota: {}, index_metrics_.cluster_size[{}]: {}", dim_, quota, level, index_metrics_.cluster_size[level]);
                    break;
                }
            }
        }
    }

    void progressive_refinement(std::vector<std::vector<std::pair<PID, size_t>>> &visited_clusters_per_layer, SearchMetrics &metrics, int refinement_level) {
        if (!refinement_budget_) {
            refinement_budget_ = std::numeric_limits<size_t>::max();
        }
        for (int l = refinement_level; l >= 1 && metrics.crack_cost < refinement_budget_; --l) {
            std::sort(visited_clusters_per_layer[l].begin(), visited_clusters_per_layer[l].end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });

            for (auto [pid, vis_cnt] : visited_clusters_per_layer[l]) {
                auto &cur_clus = cluster_at_[l][pid];
                if (!cur_clus.n1_points.empty() && vis_cnt >= early_drip_tau_ * splitting_threshold_[l]) {
                    auto con_sz = construct_nextlevel_graph_from_clus(pid, l);
                    metrics.crack_cost += kGraphInsertionCostRho * con_sz;
                    if (metrics.crack_cost >= refinement_budget_) {
                        break;
                    }
                }
            }
            if (metrics.crack_cost >= refinement_budget_) {
                break;
            }
            for (auto [pid, vis_cnt] : visited_clusters_per_layer[l]) {
                auto &cur_clus = cluster_at_[l][pid];
                if (cur_clus.n2_points.empty())
                    continue;
                if (vis_cnt > splitting_threshold_[l]) {
                    CHECK(cur_clus.n1_points.empty()) << "n1_points should be empty";
                    auto con_sz = refinement_budget_ - metrics.crack_cost;
                    metrics.crack_cost += construct_nextlevel_clus_from_clus(pid, l, con_sz);

                    if (cur_clus.n2_points.empty()) {
                        cur_clus.visited_cnt = 0;
                    }

                    if (metrics.crack_cost >= refinement_budget_) {
                        break;
                    }
                }
            }
        }

        // disable leisure cracking if duget is INF
        if (metrics.crack_cost < refinement_budget_ && (leisure_cracking_after_ >= 0 && served_query_.load() > leisure_cracking_after_)) {
            auto quota = (refinement_budget_ - metrics.crack_cost) / kGraphInsertionCostRho;
            leisure_dripping(refinement_level, quota, metrics);
        }
    }

    void construct_all_points(int num_threads) {
        BS::thread_pool pool(num_threads);

        for (int level = kMaxUnbuildLevel; level >= 1; level--) {
            // LOG(INFO) << "Constructing level " << level;
            BS::multi_future<void> futures;
            for (size_t idx = 0; idx < max_elements_; idx++) {
                if (cluster_at_[level][idx].size()) {
                    futures.push_back(pool.submit_task([idx, level, this]() {
                        construct_nextlevel_graph_from_clus(idx, level);
                    }));
                }
            }
            futures.wait();
            futures.clear();
            // LOG(INFO) << "Constructing level " << level << " done";
            for (size_t idx = 0; idx < max_elements_; idx++) {
                if (cluster_at_[level][idx].size()) {
                    futures.push_back(pool.submit_task([idx, level, this]() {
                        construct_nextlevel_clus_from_clus(idx, level);
                    }));
                }
            }
            futures.wait();
            // LOG(INFO) << "Constructing level " << level << " done";
        }
        auto clus_points_cnt = get_index_metrics().clus_points_cnt;
        if (clus_points_cnt > 0) {
            LOG(WARNING) << "Unconstruct points count: " << clus_points_cnt;
            for (size_t idx = 0; idx < max_elements_; idx++) {
                if (hang_at_ep[idx] != 0) {
                    auto [level, ep] = utils::unpack_2_uint32(hang_at_ep[idx]);
                    LOG(WARNING) << "Point " << idx << " has hang_at_ep value: (level: " << level << ", ep: " << ep << ")";
                }
            }
        }
    }

    std::vector<std::pair<PID, size_t>> probe_clusters(const FloatVec &query, int level, const buffer::SearchBufferBig &candidate_set, buffer::SearchBufferBig &next_candidate_set, SearchMetrics &metrics) {
        // Process IVF points hang at current level candidates
        auto l1 = level;
        if (nprobe_[l1] == 0 || ef_search_[l1] == 0)
            return {};

        // scan IVF
        std::vector<std::pair<PID, size_t>> visited_clusters;
        size_t probed_clusters = 0;
        candidate_set.for_each_cond([&, l1, this](const auto &entry) {
            ++probed_clusters;
            if (probed_clusters > nprobe_[l1])
                return false;

            PID pid = entry.id;
            auto &cur_clus = cluster_at_[l1][pid];

            if (cur_clus.n1_points.empty() && cur_clus.n2_points.empty())
                return true;
            auto curr_visited_cnt = ++cur_clus.visited_cnt;

            {
                visited_clusters.emplace_back(pid, curr_visited_cnt);
                auto func = [&](const auto p) {
                    if (p.point == 0)
                        return;
                    auto dist = get_distance(query, p.point);
                    metrics.dist_cmp_ivf[l1]++;
                    next_candidate_set.insert(p.point, dist);
                };
                {
                    auto &n1_p = cur_clus.n1_points;
                    std::shared_lock<std::shared_mutex> lock(n1_p.get_mutex());
                    for (size_t idx = 0; idx < n1_p.size(); idx++) {
                        func(n1_p.at_locked(idx));
                    }
                }
                {
                    auto &n2_p = cur_clus.n2_points;
                    std::shared_lock<std::shared_mutex> lock(n2_p.get_mutex());
                    for (size_t idx = 0; idx < n2_p.size(); idx++) {
                        func(n2_p.at_locked(idx));
                    }
                }
            }
            return true;
        });

        return visited_clusters;
    }
};

inline void DRIP::construct(size_t data_num, const FloatRowMat &data, int init, size_t num_threads) {
    cur_element_count_ += data_num;
    refinement_frontier_ = init;
    initial_refinement_frontier_ = init;

    utils::StopW stop;
    std::vector<size_t> labels_shuffled(data_num);
    for (size_t label = 0; label < data_num; ++label) {
        labels_shuffled[label] = label;
    }
    std::shuffle(labels_shuffled.begin(), labels_shuffled.end(), level_generator_);

    size_t assign_idx = 0;
    int assign_level = maxlevel_;
    { // assign level&pid new version
        for (assign_idx = 0; assign_idx < data_num; ++assign_idx) {
            for (; assign_level >= 0 && level_remain_sz_[assign_level] == 0; assign_level--)
                ;
            if (assign_level < 0) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }
            if (assign_level >= refinement_frontier_) {
                size_t cur_pid = level_pid_assign_[assign_level]++;
                level_remain_sz_[assign_level]--;

                auto label = labels_shuffled[assign_idx];
                label2pid_[label] = cur_pid;
                pid2label_[cur_pid] = label;

                pid2level_[cur_pid] = assign_level;
                level_lookup_[assign_level].push_back(cur_pid);
            } else {
                break;
            }
        }
    }
    auto first_label = labels_shuffled[0];
    add_new_vector(label2pid_[first_label], data.row(first_label));

    BS::thread_pool pool(num_threads);

    auto futures_void = pool.submit_loop(1, assign_idx, [this, &labels_shuffled, &data](size_t idx) {
        auto label = labels_shuffled[idx];
        auto pid = label2pid_[label];
        auto ret = add_new_vector(pid, data.row(label));
        CHECK(!ret);
    });

    futures_void.wait();
    futures_void.clear();

    LOG(INFO) << "Initial " << assign_idx << " points added in " << stop.getElapsedTimeMili() << " ms" << std::endl;
    stop.reset();

    std::vector<BS::multi_future<std::vector<std::pair<size_t, ToBeAddPoint>>>> futures_levels;
    for (size_t aidx = assign_idx, al = assign_level; aidx < data_num; --al) {
        auto next_aidx = std::min(data_num, aidx + level_remain_sz_[al]);
        auto fu = pool.submit_blocks(aidx, next_aidx, [this, &labels_shuffled, &data](size_t l, size_t r) {
            std::vector<std::pair<size_t, ToBeAddPoint>> to_be_added_points;
            to_be_added_points.reserve(r - l);
            for (size_t idx = l; idx < r; idx++) {
                auto label = labels_shuffled[idx];
                auto ret = add_new_vector(kNullPid, data.row(label));
                CHECK(ret.has_value());
                to_be_added_points.emplace_back(label, ret.value());
            }
            return to_be_added_points;
        });
        futures_levels.emplace_back(std::move(fu));
        aidx = next_aidx;
    }

    std::unordered_map<PID, std::vector<std::pair<PID, ToBeAddPoint>>> todo_clusters;
    for (auto &futures : futures_levels) {
        for (auto &f : futures) {
            auto ps = f.get();
            for (auto &p : ps) {
                // add_point_into_clus(p.c_level, p.ep, p);
                todo_clusters[p.second.ep].emplace_back(std::move(p));
            }
        }

        for (auto &[ep, to_build_points] : todo_clusters) {
            // assign level & pid
            for (auto &[label, p] : to_build_points) {
                CHECK_EQ(p.p_level, -1);
                CHECK_EQ(p.point, kNullPid);

                {
                    for (; assign_level >= 0 && level_remain_sz_[assign_level] == 0; assign_level--)
                        ;
                    if (assign_level < 0) {
                        throw std::runtime_error("The number of elements exceeds the specified limit");
                    }
                }

                size_t pid = level_pid_assign_[assign_level]++;
                level_remain_sz_[assign_level]--;
                auto level = assign_level;

                p.point = pid;
                p.p_level = level;

                label2pid_[label] = pid;
                pid2label_[pid] = label;

                pid2level_[pid] = level;
                level_lookup_[level].push_back(pid);

                add_node_into_clus(refinement_frontier_, ep, p);
            }
            futures_void.push_back(pool.submit_task([this, to_build_points = std::move(to_build_points), &data]() {
                for (auto &[label, p] : to_build_points) {
                    auto pid = p.point;
                    auto level = p.p_level;

                    // set edge data
                    set_vec_data(pid, data.row(label));
                    if (level > 0) {
                        edges_.init_up_edges(pid, level);
                        std::memcpy(edges_.get_up_vector_data(pid), data.row(label).data(), vector_data_size_);
                    }
                }
            }));
        }
        todo_clusters.clear();
    }
    LOG(INFO) << "All " << data_num << " points added in " << stop.getElapsedTimeMili() << " ms" << std::endl;
    futures_void.wait();
    futures_void.clear();

    for (int i = refinement_frontier_; i < assign_level + 1; i++) {
        index_metrics_.graph_size[i] += level_lookup_[i].size();
    }
    index_metrics_.cluster_size[refinement_frontier_] += data_num - assign_idx;
    {
        for (int l = 0; l < maxlevel_ + 1; l++) {
            LOG(INFO) << "Level " << l << " has " << level_lookup_[l].size() << " points" << std::endl;
        }
    }
}

inline std::optional<DRIP::ToBeAddPoint> DRIP::add_new_vector(PID cur_pid, const FloatVec &vec) {
    // Validate label to prevent out-of-bounds access
    if (cur_pid >= max_elements_ && cur_pid != kNullPid) {
        throw std::runtime_error("Label " + std::to_string(cur_pid) +
                                 " exceeds max_elements_ limit of " + std::to_string(max_elements_));
    }

    int curlevel = -1;
    if (cur_pid != kNullPid) {
        set_vec_data(cur_pid, vec);

        curlevel = pid2level_[cur_pid];
        // If the current vertex is at level >0, it needs some space to store the extra edges.
        if (curlevel > 0) {
            edges_.init_up_edges(cur_pid, curlevel);
            std::memcpy(edges_.get_up_vector_data(cur_pid), vec.data(), vector_data_size_);
        }
    }

    std::optional<ToBeAddPoint> ret = std::nullopt;
    PID enterpoint_node_cp = enterpoint_node_;
    int maxlevelcopy = maxlevel_;
    if (static_cast<signed>(enterpoint_node_cp) != -1) {
        ToBeAddPoint p{
            .point = cur_pid,
            .p_level = curlevel,
            .c_level = maxlevelcopy,
            .ep = enterpoint_node_cp,
            .dist = get_distance(vec, enterpoint_node_cp),
        };

        SearchMetrics metrics;
        for (; p.c_level > curlevel + 1 && p.c_level > refinement_frontier_ + 1;) {
            p.c_level--;
            find_entry_point(vec, p, metrics);
        }

        if (curlevel >= refinement_frontier_) {
            // construct the point into graph
            construct_node_2g(cur_pid, std::min(maxlevelcopy, curlevel), p.ep);
        } else {
            // lazy into ivf
            CHECK_EQ(refinement_frontier_ + 1, p.c_level);

            buffer::SearchBufferBig candidate_set(ef_assign_);
            candidate_set.insert(p.ep, p.dist);
            // find and hang under the nearest entry point
            traverse_at_level<true>(vec, refinement_frontier_, candidate_set, nullptr, metrics);

            auto lowest_entry = candidate_set.lowest_entry();
            auto curr_ep = lowest_entry.id;
            auto curr_dist = lowest_entry.distance;
            p.ep = curr_ep;
            p.c_level = refinement_frontier_;
            p.dist = curr_dist;
            CHECK_NE(curr_ep, p.point) << "hang at it self: " << p.point << " -> " << curr_ep;
            ret = p;
        }

    } else {
        // Do nothing for the first element
        enterpoint_node_ = cur_pid;
        maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_pid;
        maxlevel_ = curlevel;
    }
    return ret;
}

inline SearchResult DRIP::search_knn_cracking(const FloatVec &query, size_t topk_sz) {
    int refinement_level = refinement_frontier_.load(std::memory_order_acquire);
    if (refinement_level > 0 && index_metrics_.cluster_size[refinement_level].load() == 0) {
        std::lock_guard<std::mutex> lock(refinement_frontier_mutex_);
        refinement_level = refinement_frontier_.load(std::memory_order_relaxed);
        if (refinement_level > 0 && index_metrics_.cluster_size[refinement_level].load() == 0) {
            --refinement_level;
            leisure_position_ = 0;
            refinement_frontier_.store(refinement_level, std::memory_order_release);
        }
    }
    if (cur_element_count_ == 0) {
        LOG(ERROR) << "No elements in the index";
    }
    ++served_query_;
    SearchResult result;
    auto &metrics = result.sm;

    // Use bounded priority queue instead of the maxheap.
    std::vector<std::vector<std::pair<PID, size_t>>> visited_clusters_per_layer(refinement_level + 1);
    buffer::SearchBufferBig candidate_set(ef_search_[0]);
    buffer::SearchBufferBig result_set(topk_sz);
    {
        auto ep_id = enterpoint_node_;
        auto vl = visited_list_pool_->get_free_vislist();

        metrics.dist_cmp_graph[maxlevel_]++;
        auto est_dist = get_distance(query, ep_id);

        candidate_set.insert(ep_id, est_dist);
        vl->set(ep_id);

        for (int level = maxlevel_; level >= 0; level--) {
            // Layers completed since initialization use greedy graph routing.
            // Keep ef_search_ immutable while concurrent queries are running.
            const size_t ef = (level > refinement_level && level <= initial_refinement_frontier_)
                                  ? 1
                                  : ef_search_[level];
            candidate_set.set_capacity(std::max<size_t>(1, ef));
            traverse_at_level(query, level, candidate_set, vl.get(), metrics);
            if (level && level <= refinement_level) {
                visited_clusters_per_layer[level] = probe_clusters(query, level, candidate_set, result_set, metrics);
            }
        }

        visited_list_pool_->release_vis_list(std::move(vl));
    }
    if (refinement_budget_) {
        if (adaptive_refinement_) {
            progressive_refinement(visited_clusters_per_layer, metrics, refinement_level);
        } else {
            leisure_dripping(refinement_level, refinement_budget_, metrics, true);
        }
    }

    result_set.merge_from(candidate_set);
    result_set.for_each([&](const buffer::SearchBufferBig::Entry &entry) {
        result.results.emplace_back(entry.distance, pid2label_[entry.id]);
    });

    result.graph_candidates = std::move(candidate_set);
    return result;
}

} // namespace saqlib::drip
