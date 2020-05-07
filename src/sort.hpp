#pragma once

#include <algorithm>
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
    std::vector<long> offsets(max - min + 1, 0);

    // counts
    for (auto it = begin; it != end; ++it) {
        ++offsets[shift_key(*it)];
    }

    std::vector<long> next;
    next.reserve(offsets.size());

    // counts -> offsets, via cumsum + rightshift
    long sum = 0;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        long count = offsets[i];
        offsets[i] = sum;
        next[i] = sum;
        sum += count;
    }

    long swapped = 0;
    while (swapped != end - begin) {
        for (int_t goal = max - min - 1; goal >= 0; --goal) {
            for (long i = offsets[goal + 1] - 1; i >= next[goal]; --i) {
                std::swap(begin[i], begin[next[shift_key(begin[i])]++]);
                ++swapped;
            }
        }
    }
}

// key(list[i]) -> int \in {min...max}
// expects random access iterators
template <typename ItBeg, typename ItEnd, typename K,
          typename int_t = std::invoke_result<K, typename ItBeg::value_type>>
void sort_cleverP(ItBeg &&begin, ItEnd &&end, int_t min, int_t max, K &&key) {
    // sift_key(list[i]) -> int \in {0...max-min}
    const auto shift_key = [key = std::forward<K>(key), min](auto x) {
        return key(x) - min;
    };

    // std::size_t work = 0;
    // for (int_t goal = 0; goal < max - min; ++goal) {
    //     bool run = true;
    //     for (auto i = next[goal]; i < offsets[goal + 1]; ++i) {
    //         int_t val = shift_key(begin[i]);
    //         if (val != goal) {
    //             run = false;
    //             using std::swap;
    //             auto dest = next[val]++;
    //             swap(begin[i], begin[dest]);
    //             ++work;
    //         } else if (run) {
    //             ++next[val];
    //         }
    //     }
    // }

    // if (work == 0) {
    //     break;
    // }

    assert(min <= max);

    // zero initialise
    std::vector<int_t> parts(max - min + 1);
    std::vector<int_t> offsets(max - min + 1 + 1);
    std::vector<int_t> next(max - min + 1);

    // counts

    for (auto it = begin; it != end; ++it) {
        ++offsets[shift_key(*it)];
    }

    size_t total = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        parts[i] = i; // set goals

        size_t count = offsets[i]; // compute cumsums
        offsets[i] = total;
        next[i] = total;
        total += count;
    }
    offsets.back() = total;

    for (;;) {
        auto f = std::partition(parts.begin(), parts.end(), [&](int_t p) {
            return next[p] == offsets[p + 1];
        });

        std::size_t work = 0;
        for (auto p = f; p != parts.end() - 1; ++p) {
            auto goal = *p;
            bool run = true;
            for (auto i = next[goal]; i < offsets[goal + 1]; ++i) {
                if (int_t val = shift_key(begin[i]); val != goal) {
                    run = false;
                    using std::swap;
                    auto dest = next[val]++;
                    swap(begin[i], begin[dest]);
                    ++work;
                } else if (run) {
                    ++next[val];
                }
            }
        }

        if (work == 0) {
            break;
        }
    }
}

struct PartitionInfo {
    PartitionInfo() : count(0) {}

    union {
        size_t count;
        size_t offset;
    };
    size_t next_offset;
};

template <typename It, typename F>
inline It custom_std_partition(It begin, It end, F &&func) {
    for (;; ++begin) {
        if (begin == end)
            return end;
        if (!func(*begin))
            break;
    }
    It it = begin;
    for (++it; it != end; ++it) {
        if (!func(*it))
            continue;

        std::iter_swap(begin, it);
        ++begin;
    }
    return begin;
}

template <typename It, typename Func>
inline void unroll_loop_four_times(It begin, size_t iteration_count,
                                   Func &&to_call) {
    size_t loop_count = iteration_count / 4;
    size_t remainder_count = iteration_count - loop_count * 4;
    for (; loop_count > 0; --loop_count) {
        to_call(begin);
        ++begin;
        to_call(begin);
        ++begin;
        to_call(begin);
        ++begin;
        to_call(begin);
        ++begin;
    }
    switch (remainder_count) {
    case 3:
        to_call(begin);
        ++begin;
    case 2:
        to_call(begin);
        ++begin;
    case 1:
        to_call(begin);
    }
}

template <typename ItBeg, typename ItEnd, typename K, typename int_t>
void sort_ska(ItBeg &&begin, ItEnd &&end, int_t min, int_t max, K &&key) {

    const auto shift_key = [key = std::forward<K>(key), min](auto x) {
        return key(x) - min;
    };

    std::vector<PartitionInfo> partitions(max - min + 1);

    for (ItBeg it = begin; it != end; ++it) {
        ++partitions[shift_key(*it)].count;
    }

    std::vector<std::size_t> remaining_partitions(max - min + 1);

    size_t total = 0;
    int num_partitions = 0;

    for (decltype(max - min + 1) i = 0; i < max - min + 1; ++i) {
        size_t count = partitions[i].count;
        if (count) {
            partitions[i].offset = total;
            total += count;
            remaining_partitions[num_partitions] = i;
            ++num_partitions;
        }
        partitions[i].next_offset = total;
    }
    for (auto last_remaining = remaining_partitions.begin() + num_partitions,
              end_partition = remaining_partitions.begin() + 1;
         last_remaining > end_partition;) {
        last_remaining = custom_std_partition(
            remaining_partitions.begin(), last_remaining,
            [&](std::size_t partition) {
                size_t &begin_offset = partitions[partition].offset;
                size_t &end_offset = partitions[partition].next_offset;
                if (begin_offset == end_offset)
                    return false;

                unroll_loop_four_times(
                    begin + begin_offset, end_offset - begin_offset,
                    [&partitions, begin, &shift_key](ItBeg it) {
                        size_t this_partition = shift_key(*it);
                        size_t offset = partitions[this_partition].offset++;
                        std::iter_swap(it, begin + offset);
                    });
                return begin_offset != end_offset;
            });
    }
}

//
// // key(list[i]) -> int \in {min...max}
// // expects random access iterators
// template <typename ItBeg, typename ItEnd, typename K>
// auto sort_clever2_impl(ItBeg &&begin, ItEnd &&end, K &&key) {
//
//     auto min = key(*begin);
//     auto max = key(*begin);
//     for (auto it = begin + 1; it != end; ++it) {
//         auto k = key(*it);
//         if (k < min) {
//             min = k;
//         } else if (k > max) {
//             max = k;
//         }
//     }
//
//     using int_t = decltype(key(*begin));
//
//     // sift_key(list[i]) -> int \in {0...max-min}
//     const auto shift_key = [key = std::forward<K>(key), min](auto x) {
//         return key(x) - min;
//     };
//
//     assert(min <= max);
//
//     // zero initialise
//     std::vector<int_t> offsets(max - min + 1, 0);
//
//     // counts
//     for (auto it = begin; it != end; ++it) {
//         ++offsets[shift_key(*it)];
//     }
//
//     // counts -> offsets, via cumsum + rightshift
//     int_t sum = 0;
//     for (auto &&elem : offsets) {
//         int_t tmp = elem;
//         elem = sum;
//         sum += tmp;
//     }
//
//     // copy offsets
//
//     std::vector<int_t> next = offsets;
//
//     for (;;) {
//         std::size_t work = 0;
//         for (int_t goal = 0; goal < (int_t)next.size() - 1; ++goal) {
//             bool run = true;
//             for (auto i = next[goal]; i < offsets[goal + 1]; ++i) {
//                 if (int_t val = shift_key(begin[i]); val != goal) {
//                     run = false;
//                     using std::swap;
//                     auto dest = next[val]++;
//                     swap(begin[i], begin[dest]);
//                     ++work;
//                 } else if (run) {
//                     ++next[val];
//                 }
//             }
//         }
//
//         if (work == 0) {
//             break;
//         }
//     }
//
//     return offsets;
// }
//
// // key(list[i]) -> int \in {min...max}
// // expects random access iterators
// template <typename ItBeg, typename ItEnd, typename K, typename I>
// void sort_clever2(ItBeg &&begin, ItEnd &&end, K &&key, I min, I max) {
//
//     auto offsets = sort_clever2_impl(
//         begin, end, [&](auto x) { return (key(x) - min) * 10 / (max -
//         min); });
//
//     // for (auto it = offsets.begin(); it != offsets.end() - 1; ++it) {
//     //     if (*it != *(it + 1)) {
//     //         sort_clever2_impl(begin + *it, begin + *(it + 1), key);
//     //     }
//     // }
// }

} // namespace cj
