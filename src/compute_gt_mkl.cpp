#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <mkl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <gflags/gflags.h>

#include "data_options.h"

DEFINE_int32(gt_topk, 1000, "Number of exact nearest neighbors to write per query.");
DEFINE_int32(gt_query_batch, 4096, "Queries processed together by each MKL GEMM batch.");
DEFINE_int32(gt_base_block, 32768, "Base vectors processed by each MKL GEMM block.");
DEFINE_int32(gt_mkl_threads, 96, "MKL and OpenMP worker count.");
DEFINE_string(gt_data_dir, "", "Optional directory override for fvecs input and ivecs output.");
DEFINE_bool(gt_force, false, "Discard existing final or partial ground truth and restart.");

namespace {

using PID = uint32_t;

class MappedFvecs {
  public:
    explicit MappedFvecs(const std::string &path)
        : path_(path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Unable to open " + path + ": " + std::strerror(errno));
        }

        struct stat stat_buf {};
        if (fstat(fd_, &stat_buf) != 0) {
            const std::string error = std::strerror(errno);
            close(fd_);
            throw std::runtime_error("Unable to stat " + path + ": " + error);
        }
        size_bytes_ = static_cast<size_t>(stat_buf.st_size);
        if (size_bytes_ < sizeof(uint32_t)) {
            close(fd_);
            throw std::runtime_error(path + " is empty or has no fvecs header");
        }

        mapped_ = mmap(nullptr, size_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_ == MAP_FAILED) {
            const std::string error = std::strerror(errno);
            close(fd_);
            throw std::runtime_error("Unable to mmap " + path + ": " + error);
        }
        madvise(mapped_, size_bytes_, MADV_SEQUENTIAL);

        words_ = static_cast<const uint32_t *>(mapped_);
        dimension_ = words_[0];
        if (dimension_ == 0) {
            throw std::runtime_error(path + " has a zero-dimensional fvecs header");
        }
        record_words_ = static_cast<size_t>(dimension_) + 1;
        const size_t record_bytes = record_words_ * sizeof(uint32_t);
        if (size_bytes_ % record_bytes != 0) {
            throw std::runtime_error(path + " has an incomplete fvecs row");
        }
        rows_ = size_bytes_ / record_bytes;
    }

    MappedFvecs(const MappedFvecs &) = delete;
    MappedFvecs &operator=(const MappedFvecs &) = delete;

    ~MappedFvecs() {
        if (mapped_ != MAP_FAILED && mapped_ != nullptr) {
            munmap(mapped_, size_bytes_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    [[nodiscard]] size_t rows() const { return rows_; }
    [[nodiscard]] uint32_t dimension() const { return dimension_; }

    [[nodiscard]] const float *row(size_t index) const {
        return reinterpret_cast<const float *>(words_ + index * record_words_ + 1);
    }

    [[nodiscard]] bool header_is_valid(size_t index) const {
        return words_[index * record_words_] == dimension_;
    }

  private:
    std::string path_;
    int fd_ = -1;
    void *mapped_ = MAP_FAILED;
    size_t size_bytes_ = 0;
    const uint32_t *words_ = nullptr;
    uint32_t dimension_ = 0;
    size_t record_words_ = 0;
    size_t rows_ = 0;
};

struct Neighbor {
    float distance;
    PID id;
};

struct SmallerDistance {
    bool operator()(const Neighbor &lhs, const Neighbor &rhs) const {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.id < rhs.id;
    }
};

void add_candidate(std::vector<Neighbor> &heap, const Neighbor candidate, size_t topk) {
    if (heap.size() < topk) {
        heap.push_back(candidate);
        if (heap.size() == topk) {
            std::make_heap(heap.begin(), heap.end(), SmallerDistance{});
        }
        return;
    }

    const auto &worst = heap.front();
    if (candidate.distance > worst.distance ||
        (candidate.distance == worst.distance && candidate.id >= worst.id)) {
        return;
    }
    std::pop_heap(heap.begin(), heap.end(), SmallerDistance{});
    heap.back() = candidate;
    std::push_heap(heap.begin(), heap.end(), SmallerDistance{});
}

size_t completed_ivecs_rows(const std::filesystem::path &path, size_t topk) {
    if (!std::filesystem::exists(path)) {
        return 0;
    }
    const size_t record_bytes = (topk + 1) * sizeof(uint32_t);
    const size_t bytes = std::filesystem::file_size(path);
    if (bytes % record_bytes != 0) {
        throw std::runtime_error(path.string() + " has an incomplete ivecs row");
    }
    return bytes / record_bytes;
}

void write_metadata(const std::filesystem::path &metadata_path,
                    const std::string &dataset,
                    size_t base_rows,
                    size_t query_rows,
                    uint32_t dimension,
                    size_t topk) {
    std::ofstream metadata(metadata_path);
    if (!metadata) {
        throw std::runtime_error("Unable to write " + metadata_path.string());
    }
    metadata << "{\n"
             << "  \"dataset\": \"" << dataset << "\",\n"
             << "  \"method\": \"MKL block GEMM exact search\",\n"
             << "  \"metric\": \"L2 squared\",\n"
             << "  \"base_vectors\": " << base_rows << ",\n"
             << "  \"query_vectors\": " << query_rows << ",\n"
             << "  \"dimension\": " << dimension << ",\n"
             << "  \"top_k\": " << topk << "\n"
             << "}\n";
}

} // namespace

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_dataset.empty()) {
        throw std::runtime_error("--dataset is required");
    }
    if (FLAGS_gt_topk <= 0 || FLAGS_gt_query_batch <= 0 || FLAGS_gt_base_block <= 0 || FLAGS_gt_mkl_threads <= 0) {
        throw std::runtime_error("ground-truth top-k, block sizes, and thread count must be positive");
    }

    std::filesystem::path data_dir;
    if (FLAGS_gt_data_dir.empty()) {
        data_dir = std::filesystem::path("./data") / FLAGS_dataset;
    } else {
        data_dir = FLAGS_gt_data_dir;
    }
    const std::string query_suffix = FLAGS_query_type.empty() ? "" : "_" + FLAGS_query_type;
    const auto base_path = data_dir / (FLAGS_dataset + "_base.fvecs");
    const auto query_path = data_dir / (FLAGS_dataset + "_query" + query_suffix + ".fvecs");
    const auto gt_path = data_dir / (FLAGS_dataset + query_suffix + "_groundtruth.ivecs");
    const auto temp_path = gt_path.string() + ".tmp";
    const auto metadata_path = data_dir / ("groundtruth" + query_suffix + "_metadata.json");

    MappedFvecs base(base_path.string());
    MappedFvecs queries(query_path.string());
    if (base.dimension() != queries.dimension()) {
        throw std::runtime_error("Base and query fvecs dimensions do not match");
    }
    const size_t topk = static_cast<size_t>(FLAGS_gt_topk);
    if (topk > base.rows()) {
        throw std::runtime_error("--gt_topk exceeds the number of base vectors");
    }

    const size_t expected_gt_bytes = queries.rows() * (topk + 1) * sizeof(uint32_t);
    if (std::filesystem::exists(gt_path)) {
        if (FLAGS_gt_force) {
            std::filesystem::remove(gt_path);
        } else if (std::filesystem::file_size(gt_path) == expected_gt_bytes) {
            std::cout << "Validated existing ground truth: " << gt_path << std::endl;
            return 0;
        } else {
            throw std::runtime_error("Existing ground truth has an unexpected size: " + gt_path.string());
        }
    }
    if (FLAGS_gt_force && std::filesystem::exists(temp_path)) {
        std::filesystem::remove(temp_path);
    }

    mkl_set_dynamic(0);
    mkl_set_num_threads(FLAGS_gt_mkl_threads);
    omp_set_dynamic(0);
    omp_set_num_threads(FLAGS_gt_mkl_threads);

    std::cout << "Loading " << base.rows() << "x" << base.dimension()
              << " base vectors into contiguous memory for exact MKL search." << std::endl;
    const auto start_time = std::chrono::steady_clock::now();
    std::vector<float> base_vectors(base.rows() * static_cast<size_t>(base.dimension()));
    std::atomic<bool> invalid_base_header = false;
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < static_cast<long long>(base.rows()); ++index) {
        const size_t row = static_cast<size_t>(index);
        if (!base.header_is_valid(row)) {
            invalid_base_header.store(true, std::memory_order_relaxed);
            continue;
        }
        std::memcpy(base_vectors.data() + row * base.dimension(), base.row(row), base.dimension() * sizeof(float));
    }
    if (invalid_base_header.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Invalid fvecs header in base vectors");
    }

    std::vector<float> base_norms(base.rows());
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < static_cast<long long>(base.rows()); ++index) {
        const auto *vector = base_vectors.data() + static_cast<size_t>(index) * base.dimension();
        float norm = 0;
        for (uint32_t dim = 0; dim < base.dimension(); ++dim) {
            norm += vector[dim] * vector[dim];
        }
        base_norms[static_cast<size_t>(index)] = norm;
    }
    std::cout << "Base load finished in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() << "s." << std::endl;

    const size_t completed_rows = completed_ivecs_rows(temp_path, topk);
    if (completed_rows > queries.rows()) {
        throw std::runtime_error("Partial ground truth has more rows than the query file");
    }
    if (completed_rows) {
        std::cout << "Resuming from query row " << completed_rows << "." << std::endl;
    }
    std::ofstream output(temp_path, completed_rows ? std::ios::app | std::ios::binary : std::ios::binary);
    if (!output) {
        throw std::runtime_error("Unable to open output " + temp_path);
    }

    const size_t query_batch = static_cast<size_t>(FLAGS_gt_query_batch);
    const size_t base_block = static_cast<size_t>(FLAGS_gt_base_block);
    std::vector<float> query_vectors(query_batch * base.dimension());
    std::vector<float> query_norms(query_batch);
    std::vector<float> distances(query_batch * base_block);
    std::vector<std::unique_ptr<Neighbor[]>> block_candidate_buffers(static_cast<size_t>(FLAGS_gt_mkl_threads));
    for (auto &buffer : block_candidate_buffers) {
        buffer = std::unique_ptr<Neighbor[]>(new Neighbor[base_block]);
    }

    for (size_t query_start = completed_rows; query_start < queries.rows(); query_start += query_batch) {
        const size_t query_count = std::min(query_batch, queries.rows() - query_start);
        std::atomic<bool> invalid_query_header = false;
#pragma omp parallel for schedule(static)
        for (long long local_index = 0; local_index < static_cast<long long>(query_count); ++local_index) {
            const size_t local_row = static_cast<size_t>(local_index);
            const size_t query_row = query_start + local_row;
            if (!queries.header_is_valid(query_row)) {
                invalid_query_header.store(true, std::memory_order_relaxed);
                continue;
            }
            auto *destination = query_vectors.data() + local_row * base.dimension();
            const auto *source = queries.row(query_row);
            std::memcpy(destination, source, base.dimension() * sizeof(float));
            float norm = 0;
            for (uint32_t dim = 0; dim < base.dimension(); ++dim) {
                norm += destination[dim] * destination[dim];
            }
            query_norms[local_row] = norm;
        }
        if (invalid_query_header.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Invalid fvecs header in query vectors");
        }

        std::vector<std::vector<Neighbor>> heaps(query_count);
        for (auto &heap : heaps) {
            heap.reserve(topk);
        }

        for (size_t base_start = 0; base_start < base.rows(); base_start += base_block) {
            const size_t current_block = std::min(base_block, base.rows() - base_start);
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        static_cast<MKL_INT>(query_count),
                        static_cast<MKL_INT>(current_block),
                        static_cast<MKL_INT>(base.dimension()),
                        -2.0f,
                        query_vectors.data(),
                        static_cast<MKL_INT>(base.dimension()),
                        base_vectors.data() + base_start * base.dimension(),
                        static_cast<MKL_INT>(base.dimension()),
                        0.0f,
                        distances.data(),
                        static_cast<MKL_INT>(current_block));

#pragma omp parallel
            {
                Neighbor *block_candidates = block_candidate_buffers[static_cast<size_t>(omp_get_thread_num())].get();
#pragma omp for schedule(static)
                for (long long local_index = 0; local_index < static_cast<long long>(query_count); ++local_index) {
                    const size_t local_row = static_cast<size_t>(local_index);
                    const float query_norm = query_norms[local_row];
                    const float *distance_row = distances.data() + local_row * current_block;
                    auto &heap = heaps[local_row];
                    for (size_t local_base = 0; local_base < current_block; ++local_base) {
                        block_candidates[local_base] = Neighbor{
                            distance_row[local_base] + query_norm + base_norms[base_start + local_base],
                            static_cast<PID>(base_start + local_base)};
                    }

                    const size_t candidates_to_merge = std::min(topk, current_block);
                    if (candidates_to_merge < current_block) {
                        std::nth_element(block_candidates,
                                         block_candidates + candidates_to_merge,
                                         block_candidates + current_block,
                                         SmallerDistance{});
                    }
                    for (size_t candidate_index = 0; candidate_index < candidates_to_merge; ++candidate_index) {
                        add_candidate(heap, block_candidates[candidate_index], topk);
                    }
                }
            }
        }

        for (auto &heap : heaps) {
            if (heap.size() != topk) {
                throw std::runtime_error("Exact search returned fewer neighbors than requested");
            }
            std::sort(heap.begin(), heap.end(), SmallerDistance{});
            const uint32_t header = static_cast<uint32_t>(topk);
            output.write(reinterpret_cast<const char *>(&header), sizeof(header));
            for (const auto &neighbor : heap) {
                output.write(reinterpret_cast<const char *>(&neighbor.id), sizeof(neighbor.id));
            }
        }
        output.flush();
        if (!output) {
            throw std::runtime_error("Failed while writing " + temp_path);
        }

        const size_t completed = query_start + query_count;
        const bool is_final_batch = completed == queries.rows();
        if (is_final_batch || (completed / query_batch) % 10 == 0) {
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            const double qps = static_cast<double>(completed) / elapsed;
            std::cout << "Exact GT progress: " << completed << "/" << queries.rows() << " queries ("
                      << (100.0 * completed / queries.rows()) << "%), " << qps << " query/s." << std::endl;
        }
    }
    output.close();

    if (std::filesystem::file_size(temp_path) != expected_gt_bytes) {
        throw std::runtime_error("Final temporary ground truth has an unexpected size");
    }
    std::filesystem::rename(temp_path, gt_path);
    write_metadata(metadata_path, FLAGS_dataset, base.rows(), queries.rows(), base.dimension(), topk);
    std::cout << "Exact ground truth written to " << gt_path << " in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() << "s." << std::endl;
    return 0;
}
