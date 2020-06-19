#pragma once

#include "Eigen/Dense"
#include <cmath>
#include <iostream>

#ifndef NDEBUG
#define check(condition, message)                                              \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate();                                                  \
        }                                                                      \
    } while (false)
#else
#define check(condition, message)                                              \
    do {                                                                       \
    } while (false)
#endif

using Vector = Eigen::Array<double, Eigen::Dynamic, 1>;
using Array = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Tl, typename Tr>
inline auto dot(Tl const &v1, Tr const &v2) {
    return (v1 * v2).sum();
}

template <typename T> inline auto abs(T x) { return std::abs(x); }
