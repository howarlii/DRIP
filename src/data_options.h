#pragma once

#include <cstring>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(num_threads, 0, "number of threads to use");
DEFINE_string(data_path, "", "dataset path");
DEFINE_string(dataset, "", "dataset name.");
DEFINE_string(query_type, "", "type of query to use (e.g., skew, 10nearest)");

DEFINE_int32(searcher_dist_type, 0, "searcher distance type. 0: L2Sqr, 1: IP");

// Struct to store file paths for dataset loading
struct DataFilePaths {
    std::string input_path;
    std::string result_path;

    std::string data_file;
    std::string query_file;
    std::string gt_file;

    // Initialize file paths based on dataset parameters
    DataFilePaths(std::string method = "drip") {
        const auto &dataset = FLAGS_dataset;

        input_path = std::string("./data/") + dataset;
        result_path = fmt::format("./results/{}/", method);

        if (dataset == "mqa10m" || dataset == "wiki10m" || dataset == "laion10m") {
            std::string raw_dataname = dataset.substr(0, dataset.size() - 3);
            data_file = fmt::format("{}/{}_base.fbin", input_path, raw_dataname);
            query_file = fmt::format("{}/{}_query{}.fbin", input_path, dataset, FLAGS_query_type.empty() ? "" : ("_" + FLAGS_query_type));
        } else {
            data_file = fmt::format("{}/{}_base.fvecs", input_path, dataset);
            query_file = fmt::format("{}/{}_query{}.fvecs", input_path, dataset, FLAGS_query_type.empty() ? "" : ("_" + FLAGS_query_type));
        }

        gt_file = fmt::format("{}/{}{}_groundtruth.ivecs", input_path, dataset, FLAGS_query_type.empty() ? "" : ("_" + FLAGS_query_type));
    }
};
