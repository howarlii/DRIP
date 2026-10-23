#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#include "utils/memory.hpp"

namespace saqlib::utils {

/**
 * @brief Thread-safe concurrent vector supporting iteration, insertion, and deletion
 *
 * This implementation uses a shared_mutex for read-write synchronization:
 * - Multiple threads can read/iterate simultaneously
 * - Only one thread can write (insert/delete) at a time
 * - Write operations block all read operations temporarily
 *
 * @tparam T Element type
 * @tparam Allocator Allocator type, defaults to aligned allocator
 */
template <typename T, typename Allocator = memory::AlignedAllocator<T>>
class ConcurrentVector {
  private:
    mutable std::shared_mutex mutex_;
    std::vector<T, Allocator> data_;
    std::atomic<size_t> version_{0}; // For detecting concurrent modifications during iteration

  public:
    using value_type = T;
    using size_type = size_t;
    using reference = T &;
    using const_reference = const T &;
    using allocator_type = Allocator;

    /**
     * @brief Default constructor
     */
    ConcurrentVector() = default;

    /**
     * @brief Constructor with initial capacity
     * @param capacity Initial capacity to reserve
     */
    explicit ConcurrentVector(size_type capacity) {
        std::unique_lock lock(mutex_);
        data_.reserve(capacity);
    }

    /**
     * @brief Copy constructor
     */
    ConcurrentVector(const ConcurrentVector &other) {
        std::shared_lock other_lock(other.mutex_);
        std::unique_lock this_lock(mutex_);
        data_ = other.data_;
        version_.store(other.version_.load());
    }

    template <typename Allocator2>
    ConcurrentVector(const std::vector<T, Allocator2> &other) {
        std::unique_lock lock(mutex_);
        data_ = std::vector<T, Allocator>(other.begin(), other.end());
        version_.fetch_add(1);
    }

    /**
     * @brief Copy assignment operator
     */
    ConcurrentVector &operator=(const ConcurrentVector &other) {
        if (this != &other) {
            std::shared_lock other_lock(other.mutex_);
            std::unique_lock this_lock(mutex_);
            data_ = other.data_;
            version_.store(other.version_.load());
        }
        return *this;
    }

    /**
     * @brief Move assignment operator
     */
    ConcurrentVector &operator=(ConcurrentVector &&other) noexcept {
        if (this != &other) {
            std::unique_lock other_lock(other.mutex_);
            std::unique_lock this_lock(mutex_);
            data_ = std::move(other.data_);
            version_.store(other.version_.load());
        }
        return *this;
    }

    template <typename Allocator2>
    void merge_from(const std::vector<T, Allocator2> &other) {
        std::unique_lock this_lock(mutex_);
        data_.insert(data_.end(), other.begin(), other.end());
        version_.fetch_add(1);
    }

    auto move_and_clear_vector() {
        std::unique_lock lock(mutex_);
        auto data = std::move(data_);
        data_.clear();
        version_.fetch_add(1);
        return data;
    }

    auto &get_data_locked() { return data_; }

    std::shared_mutex &get_mutex() { return mutex_; }

    /**
     * @brief Get current size
     * @return Number of elements
     */
    size_type size() const {
        return data_.size();
    }

    /**
     * @brief Check if vector is empty
     * @return true if empty, false otherwise
     */
    bool empty() const {
        // std::shared_lock lock(mutex_);
        return data_.empty();
    }

    /**
     * @brief Get current capacity
     * @return Current capacity
     */
    size_type capacity() const {
        // std::shared_lock lock(mutex_);
        return data_.capacity();
    }

    /**
     * @brief Reserve space for at least n elements. The mutex should be locked before calling this function.
     * @param n Minimum capacity to reserve
     */
    void reserve_locked(size_type n) {
        data_.reserve(n);
        version_.fetch_add(1);
    }

    /**
     * @brief Reserve space for at least n elements
     * @param n Minimum capacity to reserve
     */
    void reserve(size_type n) {
        std::unique_lock lock(mutex_);
        reserve_locked(n);
    }

    /**
     * @brief Add element to the end. The mutex should locked before calling this function.
     * @param value Element to add
     */
    void push_back_locked(const T &value) {
        data_.push_back(value);
        version_.fetch_add(1);
    }
    /**
     * @brief Add element to the end
     * @param value Element to add
     */
    void push_back(const T &value) {
        std::unique_lock lock(mutex_);
        push_back_locked(value);
    }

    /**
     * @brief Add element to the end (move version). The mutex should be locked before calling this function.
     * @param value Element to add
     */
    void push_back_locked(T &&value) {
        data_.push_back(std::move(value));
        version_.fetch_add(1);
    }

    /**
     * @brief Add element to the end (move version)
     * @param value Element to add
     */
    void push_back(T &&value) {
        std::unique_lock lock(mutex_);
        push_back_locked(std::move(value));
    }

    /**
     * @brief Remove the last element. The mutex should be locked before calling this function.
     * @return The last element
     * @throws std::runtime_error if vector is empty
     */
    value_type pop_back_locked() {
        if (data_.empty()) {
            throw std::runtime_error("ConcurrentVector::pop_back(): vector is empty");
        }
        auto value = data_.back();
        data_.pop_back();
        version_.fetch_add(1);
        return value;
    }

    /**
     * @brief Remove the last element
     * @return The last element
     * @throws std::runtime_error if vector is empty
     */
    std::optional<value_type> pop_back() {
        std::unique_lock lock(mutex_);
        if (data_.empty()) {
            return std::nullopt;
        }
        return pop_back_locked();
    }

    /**
     * @brief Access element at index (const version)
     * @param index Index of element
     * @return Const reference to element
     * @throws std::out_of_range if index is invalid
     */
    value_type at(size_type index) const {
        std::shared_lock lock(mutex_);
        return data_.at(index);
    }

    /**
     * @brief Access element at index. The mutex should be locked before calling this function.
     * @param index Index of element
     * @return Reference to element
     * @throws std::out_of_range if index is invalid
     */
    reference at_locked(size_type index) {
        return data_.at(index);
    }

    /**
     * @brief Get first element. The mutex should be locked before calling this function.
     * @return Reference to first element
     * @throws std::runtime_error if vector is empty
     */
    reference front_locked() {
        if (data_.empty()) {
            throw std::runtime_error("ConcurrentVector::front(): vector is empty");
        }
        return data_.front();
    }

    /**
     * @brief Get first element
     * @return Reference to first element
     * @throws std::runtime_error if vector is empty
     */
    reference front() {
        std::unique_lock lock(mutex_);
        return front_locked();
    }

    /**
     * @brief Get last element (const version)
     * @return Const reference to last element
     * @throws std::runtime_error if vector is empty
     */
    const_reference back() const {
        std::shared_lock lock(mutex_);
        if (data_.empty()) {
            throw std::runtime_error("ConcurrentVector::back(): vector is empty");
        }
        return data_.back();
    }

    /**
     * @brief Get last element. The mutex should be locked before calling this function.
     * @return Reference to last element
     * @throws std::runtime_error if vector is empty
     */
    reference back_locked() {
        if (data_.empty()) {
            throw std::runtime_error("ConcurrentVector::back(): vector is empty");
        }
        return data_.back();
    }

    /**
     * @brief Get last element
     * @return Reference to last element
     * @throws std::runtime_error if vector is empty
     */
    reference back() {
        std::unique_lock lock(mutex_);
        return back_locked();
    }

    auto begin_locked() {
        return data_.begin();
    }
    auto end_locked() {
        return data_.end();
    }

    /**
     * @brief Clear all elements. The mutex should be locked before calling this function.
     */
    void clear_locked() {
        data_.clear();
        version_.fetch_add(1);
    }

    /**
     * @brief Clear all elements
     */
    void clear() {
        std::unique_lock lock(mutex_);
        clear_locked();
    }

    /**
     * @brief Resize vector to n elements. The mutex should be locked before calling this function.
     * @param n New size
     * @param value Value to initialize new elements with
     */
    void resize_locked(size_type n, const T &value = T{}) {
        data_.resize(n, value);
        version_.fetch_add(1);
    }

    /**
     * @brief Resize vector to n elements
     * @param n New size
     * @param value Value to initialize new elements with
     */
    void resize(size_type n, const T &value = T{}) {
        std::unique_lock lock(mutex_);
        resize_locked(n, value);
    }

    /**
     * @brief Erase element at specific position. The mutex should be locked before calling this function.
     * @param index Index of element to erase
     * @return true if element was erased, false if index was invalid
     */
    bool erase_locked(size_type index) {
        if (index >= data_.size()) {
            return false;
        }
        data_.erase(data_.begin() + index);
        version_.fetch_add(1);
        return true;
    }

    /**
     * @brief Erase element at specific position
     * @param index Index of element to erase
     * @return true if element was erased, false if index was invalid
     */
    bool erase(size_type index) {
        std::unique_lock lock(mutex_);
        return erase_locked(index);
    }

    bool erase(const T &value) {
        std::unique_lock lock(mutex_);
        auto it = std::find(data_.begin(), data_.end(), value);
        if (it != data_.end()) {
            data_.erase(it);
            version_.fetch_add(1);
            return true;
        }
        return false;
    }

    /**
     * @brief Thread-safe iteration over all elements
     * @param func Function to apply to each element (signature: void(const T&))
     *
     * The function holds a shared lock during the entire iteration,
     * preventing modifications but allowing concurrent reads.
     */
    template <typename Function>
    void for_each(Function &&func) const {
        std::shared_lock lock(mutex_);
        for (const auto &element : data_) {
            func(element);
        }
    }
    template <typename Function>
    void for_each_locked(Function &&func) const {
        for (const auto &element : data_) {
            func(element);
        }
    }

    /**
     * @brief Create a snapshot copy of the vector
     * @return Copy of current vector contents
     *
     * This is useful when you need to iterate over the vector
     * without holding locks for extended periods.
     */
    std::vector<T, Allocator> snapshot() const {
        std::shared_lock lock(mutex_);
        return data_;
    }

    /**
     * @brief Swap contents with another ConcurrentVector
     * @param other Other vector to swap with
     */
    void swap(ConcurrentVector &other) noexcept {
        if (this != &other) {
            std::unique_lock lock1(mutex_, std::defer_lock);
            std::unique_lock lock2(other.mutex_, std::defer_lock);
            std::lock(lock1, lock2);

            data_.swap(other.data_);
            auto this_version = version_.load();
            auto other_version = other.version_.load();
            version_.store(other_version);
            other.version_.store(this_version);
        }
    }
};

} // namespace saqlib::utils
