#pragma once

#include <algorithm> //min max
#include <cassert>
#include <cmath>
#include <cstddef> // std::size_t
#include <iostream>

#include "Eigen/Core"

#include "L_BFGS.hpp"
#include "utils.hpp"

inline constexpr double DELTA_R = 0.001;

inline constexpr int IT_MAX = 2000;
inline constexpr int IR_MAX = 10;

inline constexpr double F_TOL = 1e-8;
inline constexpr double THETA_TOL = 1 * (2 * M_PI / 360); // 1deg

inline constexpr double S_MAX = 0.5;

template <typename T, typename F>
void dimerRotate(F &grad, T &lbfgs, Vector const &R_0, Vector const &g_0,
                 Vector &g_1, Vector &N, Vector &Np, Vector &gp_1,
                 Vector &theta) {
    lbfgs.clear();
    grad(R_0 + DELTA_R * N, g_1); // test unevaluated R in grad slow-down

    for (int j = 0; j < IR_MAX; ++j) {
        lbfgs(N, g_1 - g_0, theta); // could use perpendicularised

        theta -= dot(theta, N) * N;
        theta.matrix().normalize();

        // std::cout << j << std::endl;

        double b_1 = dot(g_1 - g_0, theta) / DELTA_R;
        double c_x0 = dot(g_1 - g_0, N) / DELTA_R;
        double theta_1 = -0.5 * std::atan(b_1 / abs(c_x0));

        if (abs(theta_1) < THETA_TOL) {
            break;
        } else {
            Np = N * std::cos(theta_1) + theta * std::sin(theta_1);
            grad(R_0 + DELTA_R * Np, gp_1);

            double c_x1 = dot(gp_1 - g_0, Np) / DELTA_R;
            double a_1 = (c_x0 - c_x1 + b_1 * sin(2 * theta_1)) /
                         (1 - std::cos(2 * theta_1));
            double theta_min = 0.5 * std::atan(b_1 / a_1);

            if (a_1 * std::cos(2 * theta_min) - a_1 +
                    b_1 * std::sin(2 * theta_min) >
                0) {
                theta_min += M_PI / 2;
            }

            N = N * std::cos(theta_min) + theta * std::sin(theta_min);

            if (abs(theta_min) < THETA_TOL || j == IR_MAX - 1) {
                break;
            } else {
                g_1 = std::sin(theta_1 - theta_min) / std::sin(theta_1) * g_1 +
                      std::sin(theta_min) / std::sin(theta_1) * gp_1 +
                      (1 - std::cos(theta_min) -
                       std::sin(theta_min) * std::tan(0.5 * theta_1)) *
                          g_0;
            }
        }
    }
}

// F(R, g) -> g = gradient_at R
template <typename F> bool dimerSearch(F grad, Vector &R_0, Vector &N) {
    assert(R_0.size() == N.size());

    long dims = R_0.size();

    CoreLBFGS<6> lbfgs_rotate(dims);
    CoreLBFGS<6> lbfgs_trnslt(dims, S_MAX);

    Vector p{dims};

    Vector theta{dims};
    Vector Np{dims};

    Vector g_0{dims};
    Vector g_1{dims};
    Vector gp_1{dims};

    grad(R_0, g_0);

    for (int i = 0; i < IT_MAX; ++i) {

        // N.matrix().normalize();

        // reversing perpendicular component does not change magnitude
        if (g_0.matrix().squaredNorm() < F_TOL * F_TOL) {
            return true;
        }

        ///////////////////////////// Rotate Dimer /////////////////////////////

        dimerRotate(grad, lbfgs_rotate, R_0, g_0, g_1, N, Np, gp_1, theta);

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

        lbfgs_trnslt(R_0, g_t(), p); // grad is neg of f

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
                return false;
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
