#include <algorithm>
#include <cmath>
#include <cstring>
#include <gflags/gflags.h>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "defines.hpp"
#include "utils/BS_thread_pool.hpp"
#include "utils/IO.hpp"
#include "utils/memory.hpp"
#include "utils/pool.hpp"
#include "utils/space.hpp"

using namespace saqlib;

DEFINE_string(data_path, "", "path to dataset");
DEFINE_string(query_path, "", "output query path");
DEFINE_int32(num_centers, 10, "number of random centers");
DEFINE_int32(query_num, 1'000, "number of queries to generate (top k nearest to centers)");
DEFINE_int32(num_threads, 0, "number of threads to use (0 = auto)");
DEFINE_double(zipf_alpha, 0.0, "alpha parameter for Zipf distribution (0 = no Zipf)");
DEFINE_uint64(random_seed, 1234567, "seed for centroid selection and Zipf sampling");

struct VectorDist {
    size_t idx;
    float dist;
};

std::vector<size_t> sample_unique_zipf_ranks(size_t population, size_t sample_size, double alpha, uint64_t seed) {
    struct WeightedKey {
        double key;
        size_t rank;
    };

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), 1.0);
    std::vector<WeightedKey> keys(population);
    for (size_t rank = 0; rank < population; ++rank) {
        const double inverse_weight = std::pow(static_cast<double>(rank + 1), alpha);
        keys[rank] = {-std::log(uniform(generator)) * inverse_weight, rank};
    }

    auto by_key = [](const WeightedKey &lhs, const WeightedKey &rhs) {
        return lhs.key < rhs.key;
    };
    if (sample_size < population) {
        std::nth_element(keys.begin(), keys.begin() + sample_size, keys.end(), by_key);
    }
    keys.resize(sample_size);

    std::vector<size_t> ranks(sample_size);
    std::transform(keys.begin(), keys.end(), ranks.begin(), [](const WeightedKey &entry) {
        return entry.rank;
    });
    std::shuffle(ranks.begin(), ranks.end(), generator);
    return ranks;
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    FloatRowMat data;
    utils::load_something<float, FloatRowMat>(FLAGS_data_path.c_str(), data);

    size_t n = data.rows();
    size_t dim = data.cols();

    std::cout << "data loaded\n";
    std::cout << "\tnum_data: " << n << '\n'
              << "\tdim: " << dim << '\n';

    CHECK_GT(FLAGS_num_centers, 0) << "num_centers must be > 0";
    CHECK_LE(FLAGS_num_centers, n) << "num_centers must be <= n";
    CHECK_GT(FLAGS_query_num, 0) << "query_num must be > 0";
    CHECK_LE(FLAGS_query_num, n) << "query_num must be <= n";

    size_t thread_num = FLAGS_num_threads ? FLAGS_num_threads : 92;

    // 1. Randomly select num_centers vectors as centers
    std::mt19937_64 rng(FLAGS_random_seed);
    std::vector<size_t> all_indices(n);
    for (size_t i = 0; i < n; ++i) {
        all_indices[i] = i;
    }
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    std::vector<size_t> center_indices(all_indices.begin(), all_indices.begin() + FLAGS_num_centers);

    std::cout << "Selected " << FLAGS_num_centers << " random centers\n";

    // 2. Compute distance to nearest center for each vector (parallel)
    std::vector<VectorDist> dists(n);
    BS::thread_pool pool(thread_num);

    constexpr bool use_ip = 0;

    for (size_t i = 0; i < n; ++i) {
        pool.detach_task([&, i]() {
            float best_dist = use_ip ? -1e38f : 1e38f;

            for (size_t center_idx : center_indices) {
                float dist;
                if (use_ip) {
                    // Inner product: larger is better, so we track the maximum
                    dist = data.row(i).dot(data.row(center_idx));
                    if (dist > best_dist) {
                        best_dist = dist;
                    }
                } else {
                    // L2 distance: smaller is better
                    dist = (data.row(i) - data.row(center_idx)).squaredNorm();
                    if (dist < best_dist) {
                        best_dist = dist;
                    }
                }
            }

            dists[i] = {i, best_dist};
        });
    }

    pool.wait();
    std::cout << "Computed distances to nearest centers\n";

    // 3. Sort by distance (ascending for L2, descending for IP to get closest vectors)
    if (use_ip) {
        // For IP, larger values mean closer vectors, so sort descending
        std::sort(dists.begin(), dists.end(), [](const VectorDist &a, const VectorDist &b) {
            return a.dist > b.dist;
        });
    } else {
        // For L2, smaller distances mean closer vectors, so sort ascending
        std::sort(dists.begin(), dists.end(), [](const VectorDist &a, const VectorDist &b) {
            return a.dist < b.dist;
        });
    }

    // 4. Select top query_num vectors
    size_t query_num = std::min(static_cast<size_t>(FLAGS_query_num), n);
    FloatRowMat queries(query_num, dim);

    if (FLAGS_zipf_alpha > 0.0) {
        // Weighted sampling without replacement: rank by distance to the
        // popular centroid(s), then draw distinct ranks with Zipf weights.
        auto query_ids = sample_unique_zipf_ranks(n, query_num, FLAGS_zipf_alpha, FLAGS_random_seed + 1);

        for (size_t i = 0; i < query_num; ++i) {
            size_t selected_idx = dists[query_ids[i]].idx;
            queries.row(i) = data.row(selected_idx);
        }
        std::cout << "Selected " << query_num << " queries using Zipf distribution (alpha=" << FLAGS_zipf_alpha << ")\n";
    } else {
        // Directly take the top query_num nearest vectors
        for (size_t i = 0; i < query_num; ++i) {
            queries.row(i) = data.row(dists[i].idx);
        }
    }

    std::cout << "Selected top " << query_num << " nearest vectors as queries\n";

    // 5. Save queries
    if (FLAGS_query_path.find(".fvecs") != std::string::npos) {
        utils::save_vecs<float, FloatRowMat>(FLAGS_query_path.c_str(), queries);
        std::cout << "queries saved to " << FLAGS_query_path << '\n';
    } else if (FLAGS_query_path.find("bin") != std::string::npos) {
        utils::save_bin<float, FloatRowMat>(FLAGS_query_path.c_str(), queries);
        std::cout << "queries saved to " << FLAGS_query_path << '\n';
    } else {
        std::cerr << "Unsupported data file format: " << FLAGS_query_path << '\n';
        return -1;
    }

    return 0;
}
