#pragma once

#include "Eigen/Dense"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <utility>

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

template <typename InputIt, typename Container, typename UnaryOperation>
Container transform_into(InputIt first, InputIt last, Container &&container,
                         UnaryOperation &&unary_op) {

    std::transform(first, last, std::back_inserter(container),
                   std::forward<UnaryOperation>(unary_op));

    return container;
}

template <typename InputIt, typename UnaryOperation>
auto transform_into(InputIt first, InputIt last, UnaryOperation &&unary_op) {

    return transform_into(
        first, last,
        std::vector<std::invoke_result_t<UnaryOperation,
                                         typename InputIt::value_type>>{},
        std::forward<UnaryOperation>(unary_op));
}

bool fileExist(const std::string &name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}
