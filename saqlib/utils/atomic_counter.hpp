#pragma once

#include <atomic>

namespace saqlib::utils {

class AtomicCounter {
    using T = size_t;
    std::atomic<T> v_;

  public:
    AtomicCounter(T init_val = 0) : v_(init_val) {}
    AtomicCounter(const AtomicCounter &x) {
        v_.store(x.v_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    AtomicCounter &operator=(const AtomicCounter &x) {
        v_.store(x.v_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    T operator++() {
        return v_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    T operator++(int) {
        return v_.fetch_add(1, std::memory_order_relaxed);
    }
    T operator--() {
        return v_.fetch_sub(1, std::memory_order_relaxed) - 1;
    }
    T operator--(int) {
        return v_.fetch_sub(1, std::memory_order_relaxed);
    }

    T operator+=(T i) {
        return v_.fetch_add(i, std::memory_order_relaxed) + i;
    }
    T operator-=(T i) {
        return v_.fetch_sub(i, std::memory_order_relaxed) - i;
    }

    T operator=(T i) {
        v_.store(i, std::memory_order_relaxed);
        return i;
    }

    operator T() const { return v_.load(std::memory_order_relaxed); }
    T load() const {
        return v_.load(std::memory_order_relaxed);
    }
};

} // namespace saqlib::utils