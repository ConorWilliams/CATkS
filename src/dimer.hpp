#pragma once

#include <cassert>
#include <cmath>
#include <cstddef> // std::size_t
#include <iostream>

#include "Eigen/Core"

#include "L_BFGS.hpp"
#include "utils.hpp"

inline constexpr int IT_MAX = 2000;
inline constexpr int IR_MAX = 10;
inline constexpr double F_TOL = 0.001;
inline constexpr double DELTA_R = 0.001;
inline constexpr double S_MAX = 1;

void pp(Vector const &v) { std::cout << v.transpose() << std::endl; }

template <long M>
void dimerRotate(CoreLBFGS<M> lbfgs, Vector const &R_0, Vector const &g_0,
                 Vector &N) {
    lbfgs.clear();
    N = R_0 + g_0;
}

// F(R, g) -> g = gradient_at R
template <typename F> bool dimerSearch(Vector &R_0, Vector &N) {
    assert(R_0.size() == N.size());

    long dims = R_0.size();

    CoreLBFGS lbfgs_rotate(dims);
    CoreLBFGS lbfgs_trnslt(dims);

    F grad{};

    Vector g_0{dims};
    Vector F_T{dims};

    Vector p{dims};

    for (int i = 0; i < IT_MAX; ++i) {
        grad(R_0, g_0);

        F_T = g_0 - 2 * dot(g_0, N) * N;

        double mag = std::sqrt(dot(F_T, F_T));

        if (mag < F_TOL) {
            return true;
        }

        dimerRotate(lbfgs_rotate, R_0, g_0, N);

        ///// Translate Dimer /////

        F_T = g_0 - 2 * dot(g_0, N) * N;

        lbfgs_trnslt(R_0, -F_T, p); // grad is neg of force

        if (double s = std::sqrt(dot(p, p)); s < S_MAX) {
            R_0 -= p;
        } else {
            R_0 -= p * S_MAX / s;
        }
    }

    pp(R_0);
    pp(N);
    pp(F_T);

    return false;
}
