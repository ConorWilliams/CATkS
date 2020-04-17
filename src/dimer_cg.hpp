#pragma once

#include <algorithm> //min max
#include <cassert>
#include <cmath>
#include <cstddef> // std::size_t
#include <iostream>
#include <utility>

#include "Eigen/Core"

#include "L_BFGS.hpp"
#include "Rotor.hpp"
#include "utils.hpp"

inline constexpr int IT_MAX = 2000;
inline constexpr double F_TOL = 1e-8;
inline constexpr double S_MAX = 0.5;

// F(R, g) -> g = gradient_at R
template <typename F> bool dimerSearch(F grad, Vector &R_0, Vector &N) {
    // checks;
    assert(R_0.size() == N.size());
    N.matrix().normalize();
    // end checks

    long dims = R_0.size();

    Vector g_0{dims};
    grad(R_0, g_0);

    Rotor rotor{grad, R_0, N, g_0};

    CoreLBFGS<6> lbfgs(dims);

    //////////////////////////

    double curv = rotor.align();

    Vector gf_p{dims};
    Vector gf_n = g_0 - 2 * dot(g_0, N) * N;

    Vector p = -gf_n;

    double f0 = dot(g_0, g_0);

    constexpr double smol = 0.001;

    std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
              << std::endl;

    for (int i = 0; i < IT_MAX; ++i) {

        if (f0 < F_TOL * F_TOL) {
            return true;
        }

        f0 = dot(g_0, g_0);
        grad(R_0 + smol * p / std::sqrt(dot(p, p)), g_0);
        double fn = dot(g_0, g_0);

        if (curv > 0) {
            R_0 += 0.1 * p / std::sqrt(dot(p, p));
        } else {
            double m = (fn - f0) / smol;
            double alpha = -f0 / m;
            double norm = std::sqrt(dot(p, p));
            alpha = abs(alpha * norm) <= S_MAX ? alpha : S_MAX / norm;
            R_0 += p * alpha;
        }

        grad(R_0, g_0);

        curv = rotor.align();

        using std::swap;
        swap(gf_n, gf_p);

        gf_n = g_0 - 2 * dot(g_0, N) * N;

        double beta = dot(gf_n, gf_n - gf_p) / dot(gf_p, gf_p);

        p = -gf_n + beta * p;

        // if (curv < 0) {
        //     return true;
        // }

        std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
                  << std::endl;
    }

    return false;
}
