#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

// Hides a global for tick(), tock() timer functions
namespace utils::timer {

decltype(std::chrono::system_clock::now()) start;

} // namespace utils::timer

inline void tick() {
    utils::timer::start = std::chrono::high_resolution_clock::now();
}

template <typename... Args>
inline int tock(std::string name = "anonymous", Args &&... args) {
    constexpr int required_pad = 8;

    auto const stop = std::chrono::high_resolution_clock::now();
    auto const time = std::chrono::duration_cast<std::chrono::microseconds>(
                          stop - utils::timer::start)
                          .count();

    int const length = name.length();

    if (length < required_pad) {
        name.insert(length, required_pad - length, '-');
    }

    std::cout << name << "->" << time << " micro: ";
    (static_cast<void>(std::cout << args << ',' << ' '), ...);
    std::cout << std::endl;

    return time;
}
