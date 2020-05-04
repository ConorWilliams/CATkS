#pragma once

#include <cassert>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

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
