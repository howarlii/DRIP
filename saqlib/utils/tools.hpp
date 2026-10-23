#pragma once

#include <iomanip>
#include <queue>
#include <chrono>

#include "defines.hpp"

namespace saqlib::utils {

inline uint64_t pack_2_uint32(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32) | b;
}
inline std::pair<uint32_t, uint32_t> unpack_2_uint32(uint64_t v) {
    auto a = static_cast<uint32_t>(v >> 32);
    auto b = static_cast<uint32_t>(v & 0xFFFFFFFF);
    return {a, b};
}

inline constexpr size_t div_rd_up(size_t x, size_t y) {
    return (x / y) + static_cast<size_t>((x % y) != 0);
}

inline constexpr size_t rd_up_to_multiple_of(size_t x, size_t y) {
    return y * (div_rd_up(x, y));
}

// Helper function to parse comma-separated values
template <typename T>
inline std::vector<T> parse_comma_separated(const std::string &str) {
    std::vector<T> result;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        if (!token.empty()) {
            if constexpr (std::is_same_v<T, int>) {
                result.push_back(std::stoi(token));
            } else if constexpr (std::is_same_v<T, double>) {
                result.push_back(std::stod(token));
            }
        }
    }

    return result;
}

// Fill array with parsed values, using last value for remaining elements
template <typename T>
inline void fill_array_with_last(const std::vector<T> &values, T *array, size_t array_size) {
    if (values.empty())
        return;

    for (size_t i = 0; i < array_size; ++i) {
        if (i < values.size()) {
            array[i] = values[i];
        } else {
            array[i] = values.back(); // Use last value
        }
    }
}

inline double get_ratio(
    size_t numq,
    const FloatRowMat &query,
    const FloatRowMat &data,
    const UintRowMat &gt,
    PID *ann_results,
    size_t K,
    float (*dist_func)(const float *, const float *, size_t)) {
    std::priority_queue<float> gt_distances;
    std::priority_queue<float> ann_distances;

    for (size_t i = 0; i < K; ++i) {
        PID gt_id = gt(numq, i);
        PID ann_id = ann_results[i];
        gt_distances.emplace(dist_func(&query(numq, 0), &data(gt_id, 0), data.cols()));
        ann_distances.emplace(dist_func(&query(numq, 0), &data(ann_id, 0), data.cols()));
    }

    double ret = 0;
    size_t valid_k = 0;

    while (!gt_distances.empty()) {
        if (gt_distances.top() > 1e-5) {
            ret += std::sqrt((double)ann_distances.top() / gt_distances.top());
            ++valid_k;
        }
        gt_distances.pop();
        ann_distances.pop();
    }

    if (valid_k == 0) {
        return 1.0 * K;
    }
    return ret / valid_k * K;
}

template <typename T>
std::vector<T> horizontal_avg(const std::vector<std::vector<T>> &data) {
    size_t rows = data.size();
    size_t cols = data[0].size();

    for (auto &row : data) {
        assert(row.size() == cols);
    }

    std::vector<T> avg(cols, 0);
    for (auto &row : data) {
        for (size_t j = 0; j < cols; ++j) {
            avg[j] += row[j];
        }
    }

    for (size_t j = 0; j < cols; ++j) {
        avg[j] /= rows;
    }

    return avg;
}

inline std::string current_datetime_str() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

} // namespace saqlib::utils
