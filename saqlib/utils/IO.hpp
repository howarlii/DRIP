#pragma once

#include <algorithm>
#include <cassert>
#include <dirent.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <sys/stat.h>
#include <vector>

#include "third/Eigen/Core"

namespace saqlib::utils {
template <typename T>
void save_vector(std::ofstream &output, const std::vector<T> &vec) {
    output.write(reinterpret_cast<const char *>(&vec.size()), sizeof(size_t));
    for (const auto &item : vec) {
        output.write(reinterpret_cast<const char *>(&item), sizeof(T));
    }
}
template <typename T>
void load_vector(std::ifstream &input, std::vector<T> &vec) {
    size_t size;
    input.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    vec.clear();
    vec.resize(size);
    for (size_t i = 0; i < size; ++i) {
        input.read(reinterpret_cast<char *>(&vec[i]), sizeof(T));
    }
}

inline void save_floatvec(std::ofstream &output, const Eigen::RowVectorXf &vec) {
    size_t size = vec.size();
    output.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(vec.data()), sizeof(float) * size);
}

inline size_t load_floatvec(std::ifstream &input, Eigen::RowVectorXf &vec) {
    size_t size;
    input.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    vec.resize(size);
    input.read(reinterpret_cast<char *>(vec.data()), sizeof(float) * size);
    return size;
}

inline size_t get_filesize(const char *filename) {
    struct stat64 stat_buf;
    int rc = stat64(filename, &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

inline bool file_exists(const char *filename) {
    std::ifstream f(filename);
    if (!f.good()) {
        f.close();
        return false;
    }
    f.close();
    return true;
}

template <typename T, class M>
void load_vecs(const char *filename, M &Mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    assert(typeid(T *) == typeid(Mat.data()));

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read((char *)&tmp, sizeof(uint32_t));

    size_t cols = tmp;
    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    Mat = M(rows, cols);

    input.seekg(0, input.beg);

    for (size_t i = 0; i < rows; i++) {
        input.read((char *)&tmp, sizeof(uint32_t));
        input.read((char *)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
    input.close();
}

template <typename T, class M>
void save_vecs(const char *filename, const M &Mat) {
    std::ofstream output(filename, std::ios::binary);

    uint32_t cols = static_cast<uint32_t>(Mat.cols());
    uint32_t rows = static_cast<uint32_t>(Mat.rows());

    // Write initial cols (format header)
    // output.write(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

    // For each row, write cols followed by row data
    for (size_t i = 0; i < rows; i++) {
        output.write(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
        output.write(reinterpret_cast<const char *>(&Mat(i, 0)), sizeof(T) * cols);
    }

    output.close();
    std::cout << "File " << filename << " saved (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
}

template <typename T, class M>
void load_bin(const char *filename, M &Mat) {
    if (!file_exists(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    assert(typeid(T *) == typeid(Mat.data()));

    uint32_t rows, cols;
    std::ifstream input(filename, std::ios::binary);

    input.read((char *)&rows, sizeof(uint32_t));
    input.read((char *)&cols, sizeof(uint32_t));

    Mat = M(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        input.read((char *)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
    input.close();
}

template <typename T, class M>
void save_bin(const char *filename, const M &Mat) {
    std::ofstream output(filename, std::ios::binary);

    uint32_t rows = static_cast<uint32_t>(Mat.rows());
    uint32_t cols = static_cast<uint32_t>(Mat.cols());

    output.write(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    output.write(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

    for (size_t i = 0; i < rows; i++) {
        output.write(reinterpret_cast<const char*>(&Mat(i, 0)), sizeof(T) * cols);
    }

    output.close();
    std::cout << "File " << filename << " saved (";
    std::cout << "Rows " << rows << " Cols " << cols << ")\n"
              << std::flush;
}


// load based on file extension
template <typename T, class M>
void load_something(const std::string &filename, M &row_mat) {
    if (filename.rfind("vecs") != std::string::npos) {
        load_vecs<T, M>(filename.c_str(), row_mat);
    } else if (filename.rfind(".bin") != std::string::npos || filename.rfind(".fbin") != std::string::npos ||
               filename.rfind(".ibin") != std::string::npos) {
        load_bin<T, M>(filename.c_str(), row_mat);
    } else {
        throw std::runtime_error("Unsupported file format: " + filename);
    }
}
template <typename T, class M>
void load_something(const char *filename, M &row_mat) {
    load_something<T, M>(std::string(filename), row_mat);
}

template <typename T, class M>
void load_something_series(const std::string &filename, M &row_mat) {
    // Find the directory and prefix from filename
    size_t last_slash = filename.find_last_of("/\\");
    std::string dir = (last_slash != std::string::npos) ? filename.substr(0, last_slash) : ".";
    std::string prefix = (last_slash != std::string::npos) ? filename.substr(last_slash + 1) : filename;

    // Open directory and find all matching files
    DIR *directory = opendir(dir.c_str());
    if (!directory) {
        throw std::runtime_error("Cannot open directory: " + dir);
    }

    std::vector<std::string> matching_files;
    struct dirent *entry;
    while ((entry = readdir(directory)) != nullptr) {
        std::string entry_name(entry->d_name);
        // Check if file starts with prefix
        if (entry_name.find(prefix) == 0) {
            matching_files.push_back(dir + "/" + entry_name);
        }
    }
    closedir(directory);

    if (matching_files.empty()) {
        throw std::runtime_error("No files found with prefix: " + filename);
    }

    // Sort files to ensure consistent order
    std::sort(matching_files.begin(), matching_files.end());

    // Load all files and concatenate them
    std::vector<M> all_mats;
    size_t total_rows = 0;
    size_t cols = 0;

    for (const auto &file : matching_files) {
        M temp_mat;
        load_something<T, M>(file.c_str(), temp_mat);
        all_mats.push_back(temp_mat);
        total_rows += temp_mat.rows();
        if (cols == 0) {
            cols = temp_mat.cols();
        } else if (temp_mat.cols() != cols) {
            throw std::runtime_error("Column mismatch in file: " + file);
        }
    }

    // Allocate result matrix
    row_mat = M(total_rows, cols);

    // Copy data from all matrices
    size_t current_row = 0;
    for (const auto &mat : all_mats) {
        row_mat.block(current_row, 0, mat.rows(), mat.cols()) = mat;
        current_row += mat.rows();
    }

    std::cout << "Series loaded from " << matching_files.size() << " files (";
    std::cout << "Total Rows " << total_rows << " Cols " << cols << ")\n"
              << std::flush;
}
} // namespace saqlib::utils