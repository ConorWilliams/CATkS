#pragma once

#include "Eigen/Core"
#include <cmath>

using Vector = Eigen::Array<double, Eigen::Dynamic, 1>;
using Array = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Tl, typename Tr>
inline auto dot(Tl const &v1, Tr const &v2) {
    return (v1 * v2).sum();
}

template <typename T> inline auto abs(T x) { return std::abs(x); }
