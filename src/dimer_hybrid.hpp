#pragma once

#include <algorithm> //min max
#include <cassert>
#include <cmath>
#include <cstddef> // std::size_t
#include <iostream>

#include "Eigen/Core"

#include "L_BFGS.hpp"
#include "Rotor.hpp"
#include "utils.hpp"

inline constexpr int IT_MAX = 2000;
inline constexpr double F_TOL = 1e-8;

inline constexpr double S_MIN = 0.1;
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

    auto g_t = [&]() { return g_0 - 2 * dot(g_0, N) * N; };
    auto g_w = g_t; //[&]() { return -dot(g_0, N) * N; };

    //////////////////////////

    double curv = rotor.align();

    Vector gf_p{dims};
    Vector gf_n = g_w();

    Vector p = -gf_n;

    std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
              << std::endl;

    while (curv > 0) {
        if (dot(g_0, g_0) < F_TOL * F_TOL) {
            return true;
        }

        R_0 += S_MIN * p / std::sqrt(dot(p, p));

        grad(R_0, g_0);
        curv = rotor.align();

        using std::swap;
        swap(gf_n, gf_p);

        gf_n = g_w();

        double beta = dot(gf_n, gf_n - gf_p) / dot(gf_p, gf_p);

        p = beta * p - gf_n;

        std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
                  << std::endl;
    }

    // return false;

    grad(R_0, g_0);
    rotor.align();

    for (int i = 0; i < IT_MAX; ++i) {
        // reversing perpendicular component does not change magnitude
        if (dot(g_0, g_0) < F_TOL * F_TOL) {
            return true;
        }

        lbfgs(R_0, g_t(), p); // grad is neg of f

        double norm = std::sqrt(dot(p, p));
        double alpha = norm > S_MAX ? S_MAX / norm : 1.0;
        double phi_0 = dot(g_t(), p);

        R_0 += alpha * p;

        grad(R_0, g_0);
        rotor.align();

        double phi_a = dot(g_t(), p);

        // optional
        if (phi_a <= phi_0) {
            std::terminate();
            return false;
        }

        std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
                  << std::endl;
    }

    return false;
}
