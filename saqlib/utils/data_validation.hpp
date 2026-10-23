#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "defines.hpp"
#include "glog/logging.h"

namespace saqlib::utils {

/**
 * @brief Comprehensive data validation for FloatRowMat before HNSW index construction
 * 
 * This function performs various validation checks on the input data matrix to ensure
 * it's suitable for HNSW index construction. It checks for:
 * - Matrix dimensions and size
 * - NaN values
 * - Infinite values
 * - Data range and distribution
 * - Memory alignment
 * 
 * @param data The data matrix to validate
 * @param matrix_name Name of the matrix for logging purposes
 * @param min_dim Minimum expected dimension
 * @param max_dim Maximum expected dimension
 * @param min_rows Minimum expected number of rows
 * @param max_rows Maximum expected number of rows
 * @return true if data is valid, false otherwise
 */
inline bool validate_data_matrix(
    const FloatRowMat& data,
    const std::string& matrix_name = "data",
    size_t min_dim = 1,
    size_t max_dim = 10000,
    size_t min_rows = 1,
    size_t max_rows = 100000000) {
    
    LOG(INFO) << "Validating " << matrix_name << " matrix...";
    
    // Check matrix dimensions
    size_t rows = data.rows();
    size_t cols = data.cols();
    
    LOG(INFO) << matrix_name << " dimensions: " << rows << " x " << cols;
    
    // Validate dimensions
    if (rows == 0 || cols == 0) {
        LOG(ERROR) << matrix_name << " has zero dimensions: " << rows << " x " << cols;
        return false;
    }
    
    if (rows < min_rows) {
        LOG(ERROR) << matrix_name << " has too few rows: " << rows << " (minimum: " << min_rows << ")";
        return false;
    }
    
    if (rows > max_rows) {
        LOG(ERROR) << matrix_name << " has too many rows: " << rows << " (maximum: " << max_rows << ")";
        return false;
    }
    
    if (cols < min_dim) {
        LOG(ERROR) << matrix_name << " has too few dimensions: " << cols << " (minimum: " << min_dim << ")";
        return false;
    }
    
    if (cols > max_dim) {
        LOG(ERROR) << matrix_name << " has too many dimensions: " << cols << " (maximum: " << max_dim << ")";
        return false;
    }
    
    // Check for NaN and infinite values
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t neg_inf_count = 0;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    double sum_squares = 0.0;
    
    LOG(INFO) << "Checking for invalid values in " << matrix_name << "...";
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val = data(i, j);
            
            if (std::isnan(val)) {
                nan_count++;
                if (nan_count <= 10) {  // Log first 10 NaN locations
                    LOG(WARNING) << "NaN found at position (" << i << ", " << j << ")";
                }
            } else if (std::isinf(val)) {
                if (val > 0) {
                    inf_count++;
                    if (inf_count <= 10) {
                        LOG(WARNING) << "Positive infinity found at position (" << i << ", " << j << ")";
                    }
                } else {
                    neg_inf_count++;
                    if (neg_inf_count <= 10) {
                        LOG(WARNING) << "Negative infinity found at position (" << i << ", " << j << ")";
                    }
                }
            } else {
                // Valid number - update statistics
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum += val;
                sum_squares += val * val;
            }
        }
    }
    
    // Report invalid values
    if (nan_count > 0) {
        LOG(ERROR) << matrix_name << " contains " << nan_count << " NaN values";
        return false;
    }
    
    if (inf_count > 0) {
        LOG(ERROR) << matrix_name << " contains " << inf_count << " positive infinity values";
        return false;
    }
    
    if (neg_inf_count > 0) {
        LOG(ERROR) << matrix_name << " contains " << neg_inf_count << " negative infinity values";
        return false;
    }
    
    // Calculate and report statistics
    double mean = sum / (rows * cols);
    double variance = (sum_squares / (rows * cols)) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    LOG(INFO) << matrix_name << " statistics:";
    LOG(INFO) << "  Min value: " << min_val;
    LOG(INFO) << "  Max value: " << max_val;
    LOG(INFO) << "  Mean: " << mean;
    LOG(INFO) << "  Std deviation: " << std_dev;
    LOG(INFO) << "  Range: " << (max_val - min_val);
    
    // Check for suspicious data patterns
    if (std::abs(mean) > 1e6) {
        LOG(WARNING) << matrix_name << " has very large mean value: " << mean;
    }
    
    if (std_dev > 1e6) {
        LOG(WARNING) << matrix_name << " has very large standard deviation: " << std_dev;
    }
    
    if (max_val - min_val < 1e-10) {
        LOG(WARNING) << matrix_name << " has very small range, data might be constant";
    }
    
    // Check for constant rows (all zeros or same values)
    size_t constant_rows = 0;
    for (size_t i = 0; i < rows; ++i) {
        bool is_constant = true;
        float first_val = data(i, 0);
        for (size_t j = 1; j < cols; ++j) {
            if (std::abs(data(i, j) - first_val) > 1e-10) {
                is_constant = false;
                break;
            }
        }
        if (is_constant) {
            constant_rows++;
        }
    }
    
    if (constant_rows > 0) {
        LOG(WARNING) << matrix_name << " has " << constant_rows << " constant rows";
    }
    
    // Check memory alignment (basic check)
    if (reinterpret_cast<uintptr_t>(data.data()) % 64 != 0) {
        LOG(WARNING) << matrix_name << " data is not 64-byte aligned, may impact performance";
    }
    
    LOG(INFO) << matrix_name << " validation completed successfully";
    return true;
}

/**
 * @brief Quick validation for data matrix - only checks for critical issues
 * 
 * This is a faster version that only checks for NaN, infinity, and basic dimensions.
 * Use this when you need faster validation during performance-critical operations.
 * 
 * @param data The data matrix to validate
 * @param matrix_name Name of the matrix for logging purposes
 * @return true if data is valid, false otherwise
 */
inline bool quick_validate_data_matrix(const FloatRowMat& data, const std::string& matrix_name = "data") {
    LOG(INFO) << "Quick validation of " << matrix_name << " matrix...";
    
    size_t rows = data.rows();
    size_t cols = data.cols();
    
    if (rows == 0 || cols == 0) {
        LOG(ERROR) << matrix_name << " has zero dimensions: " << rows << " x " << cols;
        return false;
    }
    
    // Check for NaN and infinity values
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val = data(i, j);
            if (std::isnan(val)) {
                LOG(ERROR) << "NaN found at position (" << i << ", " << j << ") in " << matrix_name;
                return false;
            }
            if (std::isinf(val)) {
                LOG(ERROR) << "Infinity found at position (" << i << ", " << j << ") in " << matrix_name;
                return false;
            }
        }
    }
    
    LOG(INFO) << matrix_name << " quick validation passed";
    return true;
}

} // namespace saqlib::utils
