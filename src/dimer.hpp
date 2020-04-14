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
inline constexpr double F_TOL = 0.0001;
inline constexpr double DELTA_R = 0.001;
inline constexpr double S_MAX = 1;
inline constexpr double THETA_TOL = 0.0035; // 2deg

void pp(Vector const &v) { std::cout << v.transpose() << std::endl; }

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

    Vector theta{dims};
    Vector Np{dims};
    Vector g_1{dims};
    Vector gp_1{dims};
    Vector delta_g{dims};

    for (int i = 0; i < IT_MAX; ++i) {
        std::cout << "iteration: " << i << std::endl;
        grad(R_0, g_0);

        F_T = -g_0 + 2 * dot(g_0, N) * N;

        double mag = std::sqrt(dot(F_T, F_T));

        if (mag < F_TOL) {
            return true;
        }

        /////// Rotate Dimer ///////

        lbfgs_rotate.clear();
        grad(R_0 + DELTA_R * N, g_1);

        for (int j = 0; j < IR_MAX; ++j) {
            delta_g = g_1 - g_0;

            lbfgs_rotate(N, delta_g, theta);
            theta -= dot(theta, N) * N;
            theta.matrix().normalize();

            double b_1 = dot(delta_g, theta) / DELTA_R;
            double c_x0 = dot(delta_g, N) / DELTA_R;
            double theta_1 = -0.5 * std::atan(b_1 / abs(c_x0));

            if (abs(theta_1) < THETA_TOL) {
                break;
            } else {
                Vector p{dims};
                Np = N * std::cos(theta_1) + theta * std::sin(theta_1);
                grad(R_0 + DELTA_R * Np, gp_1);

                double c_x1 = dot(gp_1 - g_0, N) / DELTA_R;
                double a_1 = (c_x0 - c_x1 + b_1 * sin(2 * theta_1)) /
                             (1 - std::cos(2 * theta_1));
                double theta_min = 0.5 * std::atan(b_1 / a_1);

                N = N * std::cos(theta_min) + theta * std::sin(theta_min);
                if (abs(theta_min) < THETA_TOL) {
                    break;
                } else {
                    g_1 = std::sin(theta_1 - theta_min) / std::sin(theta_1) *
                              g_1 +
                          std::sin(theta_min) / std::sin(theta_1) * gp_1 +
                          (1 - std::cos(theta_min) -
                           std::sin(theta_min) * std::tan(0.5 * theta_1)) *
                              g_0;
                }
            }

            pp(N);
        }

        ////// Translate Dimer /////

        F_T = -g_0 + 2 * dot(g_0, N) * N;

        lbfgs_trnslt(R_0, -F_T, p); // grad is neg of force

        if (double s = std::sqrt(dot(p, p)); s < S_MAX) {
            R_0 -= p;
        } else {
            R_0 -= p * S_MAX / s;
        }
    }

    return false;
}
