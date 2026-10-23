#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <set>
#include <unordered_set>
#include <vector>

#include "defines.hpp"
#include "utils/memory.hpp"

namespace saqlib::buffer {
/**
 * @brief sorted linear buffer, used as beam set for graph-based ANN search.
 *
 */
class SearchBuffer {
  private:
    std::vector<Candidate, memory::AlignedAllocator<Candidate>> data_;
    size_t size_ = 0, cur_ = 0, capacity_;

    // lower_bound binary search
    [[nodiscard]] auto binary_search(float dist) const {
        size_t lo = 0;
        size_t len = size_;
        size_t half;
        while (len > 1) {
            half = len >> 1;
            len -= half;
            lo += static_cast<size_t>(data_[lo + half - 1].distance < dist) * half;
        }
        return (lo < size_ && data_[lo].distance < dist) ? lo + 1 : lo;
    }

    // set top bit to 1 as checked
    inline void set_checked(PID &data_id) {
        data_id |= (1u << 31);
    }
    inline PID unchecked_value(PID data_id) const {
        return data_id & ~(1u << 31);
    }

    [[nodiscard]] static auto is_checked(PID data_id) -> bool {
        return static_cast<bool>(data_id >> 31);
    }

  public:
    SearchBuffer() = default;

    explicit SearchBuffer(size_t capacity) : data_(capacity + 1), capacity_(capacity) {}

    // insert a data point into buffer
    void insert(PID data_id, float dist) {
        if (is_full(dist)) {
            return;
        }

        size_t lo;
        if (size_ && size_ < capacity_ && data_[size_ - 1].distance < dist) {
            lo = size_;
        } else {
            lo = binary_search(dist);
        }
        if (lo < size_ && data_[lo].id == data_id) {
            return;
        }
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Candidate));
        data_[lo] = Candidate(data_id, dist);
        size_ += static_cast<size_t>(size_ < capacity_);
        cur_ = lo < cur_ ? lo : cur_;
    }

    // get unchecked candidate with minimum distance
    PID pop() {
        PID cur_id = data_[cur_].id;
        set_checked(data_[cur_].id);
        ++cur_;
        while (cur_ < size_ && is_checked(data_[cur_].id)) {
            ++cur_;
        }
        return cur_id;
    }

    void set_all_uncheck() {
        for (size_t i = 0; i < cur_; ++i) {
            data_[i].id = unchecked_value(data_[i].id);
        }
        cur_ = 0;
    }

    void set_capacity(size_t capacity) {
        data_.resize(capacity_ + 1);
        capacity_ = capacity;
    }

    auto size() const { return size_; }

    void clear() {
        size_ = 0;
        cur_ = 0;
    }

    // return candidate id for next pop()
    [[nodiscard]] auto next_id() const { return data_[cur_].id; }
    [[nodiscard]] auto get_id(size_t idx) const { return unchecked_value(data_[idx].id); }
    [[nodiscard]] auto get_candidate(size_t idx) const {
        auto ret = data_[idx];
        ret.id = unchecked_value(ret.id);
        return ret;
    }

    [[nodiscard]] auto has_next() const -> bool { return cur_ < size_; }

    void resize(size_t new_size) {
        this->capacity_ = new_size;
        data_.resize(capacity_ + 1);
        // data_ = std::vector<Candidate, memory::AlignedAllocator<Candidate>>(
        //     capacity_ + 1);
    }

    void copy_results(PID *knn) const {
        for (size_t i = 0; i < size_; ++i) {
            knn[i] = data_[i].id;
        }
    }

    float top_dist() const {
        return (is_full() && size_ > 0) ? data_[size_ - 1].distance : std::numeric_limits<float>::max();
    }

    [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

    // judge if dist can be inserted into buffer
    [[nodiscard]] auto is_full(float dist) const -> bool { return dist > top_dist(); }

    const std::vector<Candidate, memory::AlignedAllocator<Candidate>> &data() {
        return data_;
    }
};

class SearchBufferBig {
  public:
    struct Entry {
        PID id;
        float distance;
        int label;
    };

  private:
    struct EntryComparator {
        bool operator()(const Entry &lhs, const Entry &rhs) const {
            if (lhs.distance < rhs.distance) {
                return true;
            }
            if (lhs.distance > rhs.distance) {
                return false;
            }
            if (lhs.id < rhs.id) {
                return true;
            }
            if (lhs.id > rhs.id) {
                return false;
            }
            return lhs.label < rhs.label;
        }
    };

    using CandidateSet = std::set<Entry, EntryComparator>;

    CandidateSet visited_;
    CandidateSet unvisited_;
    size_t capacity_ = std::numeric_limits<size_t>::max();
    float top_dist_cache_ = std::numeric_limits<float>::max();

    [[nodiscard]] static auto max_distance(const CandidateSet &set) -> float {
        if (set.empty()) {
            return std::numeric_limits<float>::lowest();
        }
        return set.rbegin()->distance;
    }

    void update_top_dist_cache() {
        if (!is_full() || size() == 0) {
            top_dist_cache_ = std::numeric_limits<float>::max();
        } else {
            top_dist_cache_ = std::max(max_distance(visited_), max_distance(unvisited_));
        }
    }

    void trim_excess() {
        if (capacity_ == std::numeric_limits<size_t>::max()) {
            return;
        }

        if (capacity_ == 0) {
            clear();
            return;
        }

        while (size() > capacity_) {
            bool remove_from_visited = false;

            if (visited_.empty()) {
                remove_from_visited = false;
            } else if (unvisited_.empty()) {
                remove_from_visited = true;
            } else {
                remove_from_visited = max_distance(visited_) >= max_distance(unvisited_);
            }

            if (remove_from_visited) {
                auto it = std::prev(visited_.end());
                visited_.erase(it);
            } else {
                auto it = std::prev(unvisited_.end());
                unvisited_.erase(it);
            }
        }
        update_top_dist_cache();
    }

    std::vector<Entry> sorted_entries() const {
        std::vector<Entry> entries;
        entries.reserve(size());
        entries.insert(entries.end(), visited_.begin(), visited_.end());
        entries.insert(entries.end(), unvisited_.begin(), unvisited_.end());
        std::sort(entries.begin(), entries.end(), EntryComparator{});
        return entries;
    }

  public:
    SearchBufferBig() = default;

    explicit SearchBufferBig(size_t capacity) : capacity_(capacity) {}

    void set_capacity(size_t capacity) {
        capacity_ = capacity;
        trim_excess();
        update_top_dist_cache();
    }

    void resize(size_t new_size) { set_capacity(new_size); }

    size_t size() const { return visited_.size() + unvisited_.size(); }

    bool empty() const { return size() == 0; }

    float top_dist() const { return top_dist_cache_; }

    Entry lowest_entry() const {
        if (visited_.empty()) {
            if (unvisited_.empty()) {
                return Entry{std::numeric_limits<PID>::max(), std::numeric_limits<float>::max(), -1};
            }
            return *unvisited_.begin();
        }
        if (unvisited_.empty() || visited_.begin()->distance < unvisited_.begin()->distance) {
            return *visited_.begin();
        }
        return *unvisited_.begin();
    }

    bool is_full() const { return size() >= capacity_; }

    void clear() {
        visited_.clear();
        unvisited_.clear();
        top_dist_cache_ = std::numeric_limits<float>::max();
    }

    void set_all_unvisit() {
        if (visited_.size() > unvisited_.size()) {
            std::swap(visited_, unvisited_);
            for (auto &entry : visited_) {
                unvisited_.insert(entry);
            }
            visited_.clear();
        } else {
            for (auto &entry : visited_) {
                unvisited_.insert(entry);
            }
            visited_.clear();
        }
    }

    bool has_next() const { return !unvisited_.empty(); }

    bool insert(PID data_id, float dist, int label = -1) {
        if (is_full() && dist > top_dist()) {
            return false;
        }

        Entry entry{data_id, dist, label};
        auto [_, inserted] = unvisited_.insert(entry);
        if (!inserted) {
            return false;
        }

        if (size() >= capacity_) {
            trim_excess();
            update_top_dist_cache();
        }
        return true;
    }

    void merge_from(const SearchBufferBig &other) {
        other.for_each([this](const Entry &entry) {
            insert(entry.id, entry.distance, entry.label);
        });
    }

    Entry prob_next() const {
        if (unvisited_.empty()) {
            return Entry{std::numeric_limits<PID>::max(), std::numeric_limits<float>::max(), -1};
        }
        return *unvisited_.begin();
    }

    [[nodiscard]] Entry visit_next(int target_label = -1) {
        Entry invalid{std::numeric_limits<PID>::max(), std::numeric_limits<float>::max(), -1};

        while (!unvisited_.empty()) {
            auto it = unvisited_.begin();
            Entry chosen = *it;
            visited_.insert(chosen);
            unvisited_.erase(it);

            if (target_label < 0 || it->label == target_label) {
                return chosen;
            }
        }

        return invalid;
    }

    inline void for_each_cond(const std::function<bool(const Entry &)> &callback) const {
        auto it1 = visited_.begin();
        auto it2 = unvisited_.begin();
        while (it1 != visited_.end() && it2 != unvisited_.end()) {
            if (it1->distance <= it2->distance) {
                if (!callback(*it1)) {
                    return;
                }
                ++it1;
            } else {
                if (!callback(*it2)) {
                    return;
                }
                ++it2;
            }
        }
        while (it1 != visited_.end()) {
            if (!callback(*it1)) {
                return;
            }
            ++it1;
        }
        while (it2 != unvisited_.end()) {
            if (!callback(*it2)) {
                return;
            }
            ++it2;
        }
    }

    void for_each(const std::function<void(const Entry &)> &callback) const {
        for_each_cond([callback](const Entry &entry) {
            callback(entry);
            return true;
        });
    }

    // std::vector<Entry> data() const { return sorted_entries(); }
};
} // namespace saqlib::buffer
