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
inline constexpr double F_TOL = 0.000001;
inline constexpr double DELTA_R = 0.001;
inline constexpr double S_MAX = 100;
inline constexpr double THETA_TOL = 0.0035; // 2deg

inline constexpr double C_1 = 1e-4;
inline constexpr double C_2 = 0.9;

void pp(Vector const &v) { std::cout << v.transpose() << std::endl; }

// F(R, g) -> g = gradient_at R
template <typename F> bool dimerSearch(F grad, Vector &R_0, Vector &N) {
    assert(R_0.size() == N.size());

    long dims = R_0.size();

    CoreLBFGS<5> lbfgs_rotate(dims);
    CoreLBFGS<6> lbfgs_trnslt(dims);

    Vector p{dims};

    Vector theta{dims};
    Vector Np{dims};

    Vector g_0{dims};
    Vector g_1{dims};
    Vector gp_1{dims};

    Vector buf{dims}; // scratch space

    grad(R_0, g_0);

    for (int i = 0; i < IT_MAX; ++i) {

        // N.matrix().normalize();

        // reversing perpendicular component does not change magnitude
        if (g_0.matrix().squaredNorm() < F_TOL * F_TOL) {
            return true;
        }

        ///////////////////////////// Rotate Dimer /////////////////////////////

        lbfgs_rotate.clear();
        grad(R_0 + DELTA_R * N, g_1); // test unevaluated R in grad slow-down

        for (int j = 0; j < IR_MAX; ++j) {
            lbfgs_rotate(N, g_1 - g_0, theta); // could use perpendicularised
            // theta = -theta; // maybe need theta = -theta here.
            theta -= dot(theta, N) * N;
            theta.matrix().normalize();

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
                    g_1 = std::sin(theta_1 - theta_min) / std::sin(theta_1) *
                              g_1 +
                          std::sin(theta_min) / std::sin(theta_1) * gp_1 +
                          (1 - std::cos(theta_min) -
                           std::sin(theta_min) * std::tan(0.5 * theta_1)) *
                              g_0;
                }
            }

            // pp(N);
        }

        std::cout << "R: " << R_0.transpose() << " N: " << N.transpose()
                  << " g_0: " << g_0.transpose()
                  << " g*: " << (g_0 - 2 * dot(g_0, N) * N).transpose()
                  << std::endl;

        ////////////////////////// Translate Dimer ///////////////////////////

        lbfgs_trnslt(R_0, g_0 - 2 * dot(g_0, N) * N, p); // grad is neg of f

        p = -p;

        double lo = 0;
        double hi = 1;

        double phi_0 = dot(g_0 - 2 * dot(g_0, N) * N, p);
        double phi_lo = phi_0;

        // std::cout << "p is: " << p.transpose() << std::endl;

        for (int count = 0;;) {
            grad(R_0 + hi * p, g_0); // computes & stores grad at new point

            double phi_hi = dot(g_0 - 2 * dot(g_0, N) * N, p);

            if (++count > 10) {
                hi = 1;
                std::cout << "P " << p.transpose() << std::endl;
                std::terminate();
                break;
            }

            // Wolfie 2 (strong) to garantee y^T s in bfgs is >0
            if (abs(phi_hi) <= -C_2 * phi_0) {
                R_0 += hi * p;
                break;
            }

            // std::cout << phi_0 << ' ' << lo << ' ' << hi << ' ' << phi_hi
            //           << std::endl;

            if (phi_hi >= 0) {
                // gone too far
                double r = phi_lo / phi_hi;
                hi = (hi * r - lo) / (r - 1);

            } else {
                lo = hi;
                phi_lo = phi_hi;

                hi *= 2;
            }
        }

        std::cout << "ALPHA: " << hi << std::endl;

        // if (double s = dot(p, p); s < S_MAX * S_MAX) {
        //     R_0 += p;
        // } else {
        //     R_0 += p * S_MAX / std::sqrt(s);
        // }
    }

    return false;
}
