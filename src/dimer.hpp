#pragma once

#include <cmath>
#include <cstddef> // std::size_t
#include <iostream>

#include "Eigen/Core"

#include "L_BFGS.hpp"

template <typename F> class DimerSearch {
  public:
    DimerSearch();

    DimerSearch(const DimerSearch &other) = default;
    DimerSearch(DimerSearch &&other) = default;
    DimerSearch &operator=(const DimerSearch &other) = default;
    DimerSearch &operator=(DimerSearch &&other) = default;
};
