#pragma once

#include "Eigen/Core"

using Vector = Eigen::Array<double, Eigen::Dynamic, 1>;
using Array = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

inline auto dot(Vector const &v1, Vector const &v2) { return (v1 * v2).sum(); }
