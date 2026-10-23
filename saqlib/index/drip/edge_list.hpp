#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "defines.hpp"
#include "utils/memory.hpp"
#include "utils/tools.hpp"

namespace saqlib::drip {

struct EdgeList {
    std::vector<PID *> up_edges_;
    size_t up_maxedge_;
    size_t up_size_pre_level_;

    char *up_vec_data_;
    size_t single_vec_size_;

    char *l0_edges_;
    size_t edge_offset_;
    size_t l0_size_per_element_;

    void init(size_t max_elements, size_t up_layer_size, size_t single_vec_size, size_t up_maxedge, size_t l0_maxedge) {
        up_edges_.resize(max_elements, nullptr);
        up_maxedge_ = up_maxedge;
        up_size_pre_level_ = (1 + up_maxedge);

        single_vec_size_ = utils::rd_up_to_multiple_of(single_vec_size, 64);
        up_vec_data_ = memory::align_mm<64, char>(single_vec_size_ * up_layer_size);

        auto l0_edge_size = (1 + l0_maxedge) * sizeof(PID); // (# of edges) + (edges)
        edge_offset_ = single_vec_size;
        // (raw vector data) + (# of edges) + (edges) + (padding)
        l0_size_per_element_ = utils::rd_up_to_multiple_of(single_vec_size + l0_edge_size, 64);

        l0_edges_ = memory::align_mm<64, char>(max_elements * l0_size_per_element_);
    }

    ~EdgeList() {
        for (auto &edge : up_edges_) {
            if (edge != nullptr) {
                std::free(edge);
            }
        }
        if (up_vec_data_)
            std::free(up_vec_data_);
        if (l0_edges_)
            std::free(l0_edges_);
    }

    char *get_l0_vector_data(PID pid) const {
        return l0_edges_ + (pid * l0_size_per_element_);
    }
    char *get_l0_vector_data(PID pid) {
        return l0_edges_ + (pid * l0_size_per_element_);
    }
    char *get_up_vector_data(PID pid) const {
        DCHECK_LT(pid, up_edges_.size());
        return up_vec_data_ + (pid * single_vec_size_);
    }
    char *get_up_vector_data(PID pid) {
        DCHECK_LT(pid, up_edges_.size());
        return up_vec_data_ + (pid * single_vec_size_);
    }

    void init_up_edges(size_t pid, size_t level) {
        DCHECK_GT(level, 0);
        auto up_edges_size = level * up_size_pre_level_;
        up_edges_[pid] = memory::align_mm<64, PID>(up_edges_size); // (# of edges) + (edges)
    }

    PID *get_edges(PID pid, int level) {
        if (level == 0) {
            return get_l0_edges(pid);
        }
        return (up_edges_[pid] + (level - 1) * up_size_pre_level_);
    }

    PID *get_l0_edges(PID pid) {
        return reinterpret_cast<PID *>(get_l0_vector_data(pid) + edge_offset_);
    }
};

} // namespace saqlib::drip
