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
inline constexpr double S_MAX = 0.5;

// F(R, g) -> g = gradient_at R
template <typename F> bool dimerSearch(F grad, Vector &R_0, Vector &N) {
    assert(R_0.size() == N.size());

    long dims = R_0.size();

    Vector p{dims};
    Vector g_0{dims};

    Rotor rotor{grad, R_0, N, g_0};

    CoreLBFGS<6> lbfgs(dims);

    grad(R_0, g_0);

    for (int i = 0; i < IT_MAX; ++i) {

        // N.matrix().normalize();

        // reversing perpendicular component does not change magnitude
        if (g_0.matrix().squaredNorm() < F_TOL * F_TOL) {
            return true;
        }

        ///////////////////////////// Rotate Dimer /////////////////////////////

        rotor.align();

        // std::cout << "R: " << R_0.transpose() << " N: " << N.transpose()
        //           << " g_0: " << g_0.transpose()
        //           << " g*: " << (g_0 - 2 * dot(g_0, N) * N).transpose()
        //           << std::endl;

        // std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
        //           << ' ' << R_0(2) << ' ' << R_0(3) << std::endl;

        std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
                  << std::endl;

        ////////////////////////// Translate Dimer /////////////////////////////

        auto g_t = [&]() { return g_0 - 2 * dot(g_0, N) * N; };

        lbfgs(R_0, g_t(), p); // grad is neg of f

        double norm = std::sqrt(dot(p, p));

        double alpha = norm > S_MAX ? S_MAX / norm : 1.0;

        double phi_0 = dot(g_t(), p);

        // double mag_0 = dot(g_0, g_0);

        constexpr double mult = 1.5;

        grad(R_0 + alpha * p, g_0);

        while (alpha < S_MAX / norm / mult) {

            // double mag_a = dot(g_0, g_0);

            double phi_a = dot(g_t(), p);

            if (phi_a > phi_0) {
                break;
            } else {
                // return false;
                // std::cout << "p" << std::endl;
                alpha *= mult;
                grad(R_0 + alpha * p, g_0);
            }
        }

        R_0 += alpha * p;
    }

    return false;
}
