#pragma once

#include <cassert>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace cj {

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename ItBeg, typename ItEnd, typename I>
inline void sort(ItBeg &&begin, ItEnd &&end, I min, I max) {
    sort(std::forward<ItBeg>(begin), std::forward<ItEnd>(end), min, max,
         [](auto x) { return x; });
}

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename ItBeg, typename ItEnd, typename K,
          typename int_t = std::invoke_result<K, typename ItBeg::value_type>>
void sort(ItBeg &&begin, ItEnd &&end, int_t min, int_t max, K &&key) {
    // sift_key(list[i]) -> int \in {0...max-min}
    const auto shift_key = [key = std::forward<K>(key), min](auto x) {
        return key(x) - min;
    };

    assert(min <= max);

    // zero initialise
    std::vector<int_t> offsets(max - min + 1, 0);

    // counts
    for (auto it = begin; it != end; ++it) {
        ++offsets[shift_key(*it)];
    }

    // counts -> offsets, via cumsum + rightshift
    int_t sum = 0;
    for (auto &&elem : offsets) {
        int_t tmp = elem;
        elem = sum;
        sum += tmp;
    }

    // copy offsets
    std::vector<int_t> next = offsets;

    int_t i = 0, goal = 0;
    while (goal < max - min) { // can skip last block
        if (i >= offsets[goal + 1]) {
            // increase goal if in next block
            ++goal;
        } else {
            int_t val = shift_key(begin[i]);
            if (val == goal) {
                // no swap needed if value in goal region
                ++i;
            } else {
                // swap to final location
                int_t dest = next[val]++; // ++ keeps track
                using std::swap;
                swap(begin[i], begin[dest]);
            }
        }
    }
}

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename T, typename int_t>
T sort2(T const &list, int_t min, int_t max) {
    return sort2(list, min, max, [](auto x) { return x; });
}

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename T, typename K,
          typename int_t = std::invoke_result<K, typename T::value_type>>
T sort2(T const &list, int_t min, int_t max, K &&key) {
    // sift_key(list[i]) -> int \in {0...max-min}
    auto const shift_key = [&](auto x) { return key(x) - min; };

    assert(min <= max);

    // zero initialise
    std::vector<int_t> offsets(max - min + 1, 0);

    // counts
    for (auto const &elem : list) {
        ++offsets[shift_key(elem)];
    }

    // counts -> offsets, via cumsum + rightshift
    int_t sum = 0;
    for (auto &&elem : offsets) {
        int_t tmp = elem;
        elem = sum;
        sum += tmp;
    }

    // new
    T sorted(list.size());

    for (auto const &elem : list) {
        sorted[offsets[shift_key(elem)]++] = elem;
    }

    return sorted;
}

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename ItBeg, typename ItEnd, typename I>
inline void sort_clever(ItBeg &&begin, ItEnd &&end, I min, I max) {
    sort_clever(std::forward<ItBeg>(begin), std::forward<ItEnd>(end), min, max,
                [](auto x) { return x; });
}

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename ItBeg, typename ItEnd, typename K,
          typename int_t = std::invoke_result<K, typename ItBeg::value_type>>
void sort_clever(ItBeg &&begin, ItEnd &&end, int_t min, int_t max, K &&key) {
    // sift_key(list[i]) -> int \in {0...max-min}
    const auto shift_key = [key = std::forward<K>(key), min](auto x) {
        return key(x) - min;
    };

    assert(min <= max);

    // zero initialise
    std::vector<int_t> offsets(max - min + 1, 0);

    // counts
    for (auto it = begin; it != end; ++it) {
        ++offsets[shift_key(*it)];
    }

    // counts -> offsets, via cumsum + rightshift
    int_t sum = 0;
    for (auto &&elem : offsets) {
        int_t tmp = elem;
        elem = sum;
        sum += tmp;
    }

    // copy offsets

    std::vector<int_t> next = offsets;

    for (;;) {
        std::size_t work = 0;
        for (int_t goal = 0; goal < (int_t)next.size() - 1; ++goal) {
            for (auto i = next[goal]; i < offsets[goal + 1]; ++i) {
                // std::cout << i << ' ';
                int_t val = shift_key(begin[i]);
                if (val != goal) {
                    using std::swap;
                    // swap to final location
                    auto dest = next[val]++;
                    swap(begin[i], begin[dest]);
                    ++work;
                }
            }
        }

        // std::cout << "woof " << work << std::endl;

        if (work == 0) {
            break;
        }
    }
}

} // namespace cj
