// HNSW is developed from the [HNSW library](https://github.com/nmslib/hnswlib)
#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "third/Eigen/src/Core/Map.h"
#include "utils/buffer.hpp"
#include "utils/memory.hpp"
#include "utils/visited_pool.hpp"

#include "edge_list.hpp"

namespace saqlib::drip {

static constexpr size_t kMaxLevel = 8;

struct SearchMetrics {
    std::array<size_t, kMaxLevel> visited_nodes{};
    std::array<size_t, kMaxLevel> dist_cmp_graph{};
    std::array<size_t, kMaxLevel> dist_cmp_ivf{};
    size_t crack_cost{0};

    void accumulate(const SearchMetrics &other) {
        for (size_t i = 0; i < kMaxLevel; i++) {
            visited_nodes[i] += other.visited_nodes[i];
            dist_cmp_graph[i] += other.dist_cmp_graph[i];
            dist_cmp_ivf[i] += other.dist_cmp_ivf[i];
        }
    }
};

class HNSW {
  public:
    explicit HNSW() = default;
    explicit HNSW(
        size_t num_dim, size_t max_elements, size_t M, size_t ef_build,
        size_t random_seed = 100, DistType metric_type = DistType::L2Sqr);
    ~HNSW() = default;

    auto get_max_level() const { return maxlevel_; }
    size_t get_max_elements() const { return max_elements_; }
    size_t get_current_element_count() const { return cur_element_count_; }

    void construct(size_t data_num, const FloatRowMat &data, size_t num_threads = 0);

  protected:
    static constexpr PID kMaxLabelOperationLock = 65536;
    static constexpr PID kNullPid = std::numeric_limits<PID>::max();

    struct ToBeAddPoint {
        PID point = 0;
        int p_level = 0;
        int c_level = 0;
        PID ep = 0;
        float dist = 0;

        bool operator==(const ToBeAddPoint &other) const {
            return point == other.point && c_level == other.c_level;
        }
    };

    size_t max_elements_{0};
    size_t dim_{0};
    size_t vector_data_size_{0}; // Raw vector bytes stored per element.
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0}; // Fixed highest layer H in the paper.
    DistType metric_type_;
    float (*raw_dist_func_)(const FloatVec &a, const FloatVec &b);
    std::default_random_engine level_generator_;

    size_t ef_build_{0}; // HNSW construction beam width ef_build.
    PID enterpoint_node_{0};
    std::atomic<size_t> cur_element_count_{1}; // current number of elements

    std::array<size_t, kMaxLevel + 1> level_pid_assign_;
    std::array<size_t, kMaxLevel + 1> level_remain_sz_;
    std::array<std::vector<PID>, kMaxLevel + 1> level_lookup_; // {level -> pid list}
    std::vector<int> pid2level_;                               // {pid -> level)
    std::vector<PID> pid2label_;                               // {pid -> external label}
    std::unordered_map<PID, PID> label2pid_;                   // {external label -> pid}

    EdgeList edges_;
    std::vector<std::shared_mutex> link_list_locks_;

    // ===== searching related =====
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // std::mutex global_; //////// why move above will slow down ????

    int get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        // r = std::min(r, static_cast<double>(maxlevel_limit_));
        return static_cast<int>(r);
    }

    std::pair<PID, int> get_new_pid() {
        int curr_level = get_random_level();
        for (; curr_level >= 0 && level_remain_sz_[curr_level] == 0; curr_level--)
            ;
        if (curr_level < 0) {
            for (int l = 0; l <= maxlevel_; l++)
                if (level_remain_sz_[l] > 0) {
                    curr_level = l;
                    break;
                }
        }
        if (curr_level < 0) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }
        level_remain_sz_[curr_level]--;
        size_t cur_pid = level_pid_assign_[curr_level]++;
        return {cur_pid, curr_level};
    }

    void set_vec_data(PID pid, const FloatVec &vec) {
        std::memcpy(edges_.get_l0_vector_data(pid), vec.data(), dim_ * sizeof(float));
    }

    Eigen::Map<const FloatVec> get_up_vec_data(PID pid) const {
        return Eigen::Map<const FloatVec>(reinterpret_cast<const float *>(edges_.get_up_vector_data(pid)), dim_);
    }
    Eigen::Map<const FloatVec> get_l0_vec_data(PID pid) const {
        return Eigen::Map<const FloatVec>(reinterpret_cast<const float *>(edges_.get_l0_vector_data(pid)), dim_);
    }

    float get_data_dist(PID pid1, PID pid2) {
        return raw_dist_func_(get_l0_vec_data(pid1), get_l0_vec_data(pid2));
    }

    float get_distance(const FloatVec &query, PID pid) {
        return raw_dist_func_(query, get_l0_vec_data(pid));
    }

    void find_entry_point(const FloatVec &query, ToBeAddPoint &p, SearchMetrics &metrics) {
        DCHECK_GE(p.ep, pid2level_[p.point]);
        bool changed = true;
        while (changed) {
            changed = false;
            // std::shared_lock<std::shared_mutex> lock(link_list_locks_[p.ep]); // !!!!!!!!!!!!!!!!!!!!!
            PID *data = edges_.get_edges(p.ep, p.c_level);
            PID size = *data;

            auto *datal = data + 1;
            for (PID i = 0; i < size; i++) {
                PID cand = datal[i];
                if (cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }
                float d = raw_dist_func_(query, get_up_vec_data(cand));
                if (d < p.dist) {
                    p.dist = d;
                    p.ep = cand;
                    changed = true;
                }
            }
            metrics.visited_nodes[p.c_level]++;
            metrics.dist_cmp_graph[p.c_level] += size;
        }
    }

    void construct_node_2g(PID cur_point, int level, PID ep) {
        SearchMetrics metrics;
        auto vl = visited_list_pool_->get_free_vislist();
        auto curr_vec = get_l0_vec_data(cur_point);
        buffer::SearchBufferBig candidate_set(ef_build_);
        candidate_set.insert(ep, get_data_dist(ep, cur_point));
        vl->set(ep);
        {
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[cur_point]);
            for (auto l = level; l >= 0; l--) {
                traverse_at_level(curr_vec, l, candidate_set, vl.get(), metrics);
                mutually_connect_new_element(cur_point, candidate_set, l);
            }
        }
        visited_list_pool_->release_vis_list(std::move(vl));
    }

    template <bool is_upper_level = false>
    void traverse_at_level(const FloatVec &query, int level, buffer::SearchBufferBig &candidate_set, HashBasedBooleanSet *vl, SearchMetrics &metrics) {
        if (is_upper_level) {
            CHECK_GT(level, 0);
        }
        candidate_set.set_all_unvisit();
        std::unique_ptr<HashBasedBooleanSet> vl_carried;
        if (vl == nullptr) {
            vl_carried = visited_list_pool_->get_free_vislist();
            vl = vl_carried.get();
        }
        while (candidate_set.has_next()) {
            metrics.visited_nodes[level]++;

            // Step 1 - get the next node to explore.
            PID current_node_id = candidate_set.visit_next().id;
            // std::shared_lock<std::shared_mutex> lock(link_list_locks_[current_node_id]); // lock for reading, optional
            DCHECK_LE(level, pid2level_[current_node_id]);
            PID *data = edges_.get_edges(current_node_id, level);
            PID size = *data;

            // Iterate over neighbors. (List starts at index 1.)
            for (size_t j = 1; j <= size; j++) {
                PID candidate_id = *(data + j);

                if (j < size) {
                    if (is_upper_level) {
                        memory::mem_prefetch_l1(edges_.get_up_vector_data(*(data + j + 1)), vector_data_size_ / 64);
                    } else {
                        memory::mem_prefetch_l1(edges_.get_l0_vector_data(*(data + j + 1)), vector_data_size_ / 64);
                    }
                }

                if (!vl->get(candidate_id)) {
                    vl->set(candidate_id);
                    // CHECK_GT(element_levels_.size(), candidate_id);
                    DCHECK_GE(pid2level_[candidate_id], level) << "Element " << candidate_id << " is at level " << pid2level_[candidate_id] << " but visited at " << level << '\n';

                    auto dist = raw_dist_func_(query, is_upper_level ? get_up_vec_data(candidate_id) : get_l0_vec_data(candidate_id));
                    metrics.dist_cmp_graph[level]++;

                    candidate_set.insert(candidate_id, dist);
                }
            }
        }
        if (vl_carried) {
            visited_list_pool_->release_vis_list(std::move(vl_carried));
        }
    }

    std::vector<std::pair<float, PID>> get_neighbors_by_heuristic2(
        buffer::SearchBufferBig &candidates, size_t M) {
        std::vector<std::pair<float, PID>> return_list;
        return_list.reserve(M);

        if (candidates.size() < M) {
            candidates.for_each([&](const auto &entry) {
                PID id = entry.id;
                float dist = entry.distance;
                return_list.emplace_back(dist, id);
            });
            return return_list;
        }

        candidates.for_each_cond([&, this](const auto &entry) {
            PID id = entry.id;
            float dist = entry.distance;
            bool good = true;

            for (std::pair<float, PID> second_pair : return_list) {
                float curdist = get_data_dist(second_pair.second, id);
                if (curdist < dist) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.emplace_back(dist, id);
            }
            return return_list.size() < M;
        });
        return return_list;
    }

    PID mutually_connect_new_element(PID cur_p, buffer::SearchBufferBig &candidates, int level) {
        size_t max_m = level > 0 ? maxM_ : maxM0_;
        auto selected_pairs = get_neighbors_by_heuristic2(candidates, M_);
        if (selected_pairs.size() > M_) {
            throw std::runtime_error(
                "Should be not be more than M_ candidates returned by the heuristic");
        }

        // {
        //     std::string str;
        //     for (auto sn : selected_pairs) {
        //         str += std::to_string(sn.second) + " ";
        //     }
        //     LOG(INFO) << "Level / point : " << level << " / " << cur_p << " with # of neighbors: "
        //               << selected_pairs.size() << " : " << str;
        // }

        PID next_closest_entry_point = selected_pairs.front().second;

        {
            PID *ll_cur = edges_.get_edges(cur_p, level);

            if (*ll_cur > 0) {
                LOG(WARNING) << "The newly inserted element should have blank link list.  point " << cur_p << " level " << level << " and size " << *ll_cur;
                throw std::runtime_error(
                    "The newly inserted element should have blank link list");
            }

            auto *data = static_cast<PID *>(ll_cur + 1);
            for (size_t idx = 0; idx < selected_pairs.size(); idx++) {
                CHECK_EQ(data[idx], 0) << "data[idx] is not 0, Possible memory corruption";
                DCHECK_LE(level, pid2level_[selected_pairs[idx].second]) << "Trying to make a link on a non-existent level: " << level << " > " << pid2level_[selected_pairs[idx].second] << " selected_neighbor: " << selected_pairs[idx].second << " cur_p: " << cur_p;

                data[idx] = selected_pairs[idx].second;
            }

            *ll_cur = selected_pairs.size();
        }

        for (auto [dist, selected_neighbor] : selected_pairs) {
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[selected_neighbor], std::defer_lock);
            if (!lock.try_lock()) {
                continue;
            }

            PID *ll_other = edges_.get_edges(selected_neighbor, level);
            PID sz_link_list_other = *ll_other;

            // if (hang_at_ep[selected_neighbor] != 0) {
            //     auto [hl, hep] = utils::unpack_2_uint32(hang_at_ep[selected_neighbor]);
            //     // LOG(WARNING) << "Point " << selected_neighbor << fmt::format(" at {}/{}; hang_at_ep do not match the value. should be {}/{}, but is {}/{}.", selected_neighbor, level, curr_level, curr_ep);
            //     CHECK_EQ(hang_at_ep[selected_neighbor], 0) << fmt::format("selected_neighbor: {}/{} hang at {}/{}, cur_p: {}/{}", selected_neighbor, pid2level_[selected_neighbor], hep, hl, cur_p, level);
            // }

            // if (selected_neighbor != enterpoint_node_ && sz_link_list_other == 0) {
            //     CHECK_GT(sz_link_list_other, 0) << "sz_link_list_other: " << sz_link_list_other << " selected_neighbor: " << selected_neighbor << " level: " << level;
            //     throw std::runtime_error("selected_neighbor is not the enterpoint_node_ and sz_link_list_other is not 0");
            // }

            if (sz_link_list_other > max_m) {
                throw std::runtime_error("Bad value of sz_link_list_other");
            }
            if (selected_neighbor == cur_p) {
                throw std::runtime_error("Trying to connect an element to itself");
            }
            if (level > pid2level_[selected_neighbor]) {
                LOG(ERROR) << "Trying to make a link on a non-existent level: "
                           << level << " > " << pid2level_[selected_neighbor] << " selected_neighbor: " << selected_neighbor << " cur_p: " << cur_p;
                throw std::runtime_error("Trying to make a link on a non-existent level");
            }

            auto *data = static_cast<PID *>(ll_other + 1);

            bool is_cur_c_present = false;
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_p) {
                    is_cur_c_present = true;
                    break;
                }
            }

            if (!is_cur_c_present) {
                if (sz_link_list_other < max_m) {
                    data[sz_link_list_other] = cur_p;
                    *ll_other = sz_link_list_other + 1;
                } else {
                    buffer::SearchBufferBig candidates(max_m + 1);
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.insert(data[j], get_data_dist(data[j], selected_neighbor));
                    }
                    candidates.insert(cur_p, get_data_dist(selected_neighbor, cur_p));

                    auto return_list = get_neighbors_by_heuristic2(candidates, M_);
                    int indx = 0;
                    for (auto [dist, id] : return_list) {
                        data[indx] = id;
                        indx++;
                    }
                    *ll_other = indx;
                }
            }
        }

        return next_closest_entry_point;
    }
};

inline HNSW::HNSW(
    size_t num_dim,
    size_t max_elements,
    size_t M,
    size_t ef_build,
    size_t random_seed,
    DistType metric_type)
    : max_elements_(max_elements),
      dim_(num_dim),
      metric_type_(metric_type),
      pid2level_(max_elements),
      pid2label_(max_elements),
      link_list_locks_(max_elements) {
    // Set distance function based on metric type
    if (metric_type == DistType::IP) {
        raw_dist_func_ = [](auto a, auto b) -> float {
            return a.dot(b);
        };
    } else {
        raw_dist_func_ = [](auto a, auto b) -> float {
            return (a - b).squaredNorm();
        };
    }

    // graph parameters
    enterpoint_node_ = -1;
    CHECK_LT(M, 1000) << "M parameter is too large";
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_build_ = std::max(ef_build, M_);
    vector_data_size_ = num_dim * sizeof(float);
    edges_.init(max_elements, max_elements / M_ + 1, vector_data_size_, maxM_, maxM0_);
    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    // level generation
    level_generator_.seed(random_seed);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));
    revSize_ = 1.0 / mult_;
    level_pid_assign_.fill(0);
    maxlevel_ = 0;
    for (size_t curr_sz = max_elements / M_; curr_sz > 0; curr_sz /= M_, maxlevel_++) {
        level_pid_assign_[maxlevel_] = curr_sz + 1;
    }
    level_pid_assign_[maxlevel_] = 1;
    for (int l = maxlevel_; l > 0; l--) {
        level_remain_sz_[l] = level_pid_assign_[l - 1] - level_pid_assign_[l];
    }
    level_remain_sz_[0] = max_elements - level_pid_assign_[0] + 1;
}

} // namespace saqlib::drip
