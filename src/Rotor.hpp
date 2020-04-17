#pragma once

#include <cmath>

#include "Eigen/Core"
#include "L_BFGS.hpp"
#include "utils.hpp"

inline constexpr double DELTA_R = 0.001;
inline constexpr int IR_MAX = 10;
inline constexpr double THETA_TOL = 1 * (2 * M_PI / 360); // 1deg

template <typename F> class Rotor {
    F &grad;

    Vector const &R_0;
    Vector &N;
    Vector const &g_0;

    CoreLBFGS<3> lbfgs;

    Vector g_1;
    Vector gp_1;
    Vector Np;
    Vector theta;

  public:
    Rotor(F &grad, Vector const &R_0, Vector &N, Vector const &g_0)
        : grad{grad}, R_0{R_0}, N{N}, g_0{g_0}, lbfgs(R_0.size()),
          g_1{R_0.size()}, gp_1{R_0.size()}, Np{R_0.size()}, theta{R_0.size()} {
    }

    inline double align() {
        lbfgs.clear();
        grad(R_0 + DELTA_R * N, g_1); // test unevaluated R in grad slow-down

        for (int iter = 0;; ++iter) {
            lbfgs(N, g_1 - g_0, theta); // could use perpendicularised

            theta -= dot(theta, N) * N;
            theta.matrix().normalize();

            // std::cout << j << std::endl;

            double b_1 = dot(g_1 - g_0, theta) / DELTA_R;
            double c_x0 = dot(g_1 - g_0, N) / DELTA_R;
            double theta_1 = -0.5 * std::atan(b_1 / abs(c_x0));

            if (abs(theta_1) < THETA_TOL || iter == IR_MAX) {
                return c_x0;
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

                g_1 = std::sin(theta_1 - theta_min) / std::sin(theta_1) * g_1 +
                      std::sin(theta_min) / std::sin(theta_1) * gp_1 +
                      (1 - std::cos(theta_min) -
                       std::sin(theta_min) * std::tan(0.5 * theta_1)) *
                          g_0;

                if (abs(theta_min) < THETA_TOL) {
                    return dot(g_1 - g_0, N) / DELTA_R;
                }
            }
        }
    }
};
