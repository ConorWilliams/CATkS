#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "Eigen/Core"
#include "L_BFGS.hpp"
#include "utils.hpp"

inline constexpr int IR_MAX = 10;
inline constexpr int IE_MAX = 100;
inline constexpr int IT_MAX = 1950;

inline constexpr double DELTA_R = 0.001;

inline constexpr double F_TOL = 1e-8;
inline constexpr double THETA_TOL = 5 * (2 * M_PI / 360); // 1deg

inline constexpr double S_MIN = 0.1;
inline constexpr double S_MAX = 0.5;

template <typename F> class Dimer {
    F const &grad;
    long dims;

    CoreLBFGS<4> lbfgs_rot;
    CoreLBFGS<6> lbfgs_trn;

    Vector &R_0;
    Vector &N;

    Vector g_0;

    // shared space
    Vector s1;
    Vector s2;
    Vector s3;

    // rotation storage
    Vector g_1;
    Vector gp_1;
    Vector g_delta;
    Vector Np;
    Vector theta;

  public:
    Dimer(F const &grad, Vector &R_0, Vector &N)
        : grad{grad}, dims{N.size()}, lbfgs_rot(dims),
          lbfgs_trn(dims), R_0{R_0}, N{N}, g_0{dims}, s1{dims}, s2{dims},
          s3{dims}, g_1{dims}, gp_1{dims}, Np{dims}, theta{dims} {
        assert(R_0.size() == N.size());
        updateGrad();
    }

    inline double alignAxis() {

        lbfgs_rot.clear();
        grad(R_0 + DELTA_R * N, g_1); // test unevaluated R in grad slow-down

        for (int iter = 0;; ++iter) {
            g_delta = g_1 - g_0;
            lbfgs_rot(N, g_delta, theta); // could use perpendicularised

            theta -= dot(theta, N) * N;
            theta.matrix().normalize();

            double b_1 = dot(g_delta, theta) / DELTA_R;
            double c_x0 = dot(g_delta, N) / DELTA_R;
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

    inline void updateGrad() { grad(R_0, g_0); }

    template <typename T> inline void translate(T const &x) {
        print();
        R_0 += x;
        updateGrad();
    }

    inline auto effGrad() const { return g_0 - 2 * dot(g_0, N) * N; }

    void print() const {
        std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
                  << std::endl;
    }

    bool escapeConvex() {
        Vector &gf_p = s1;
        Vector &gf_n = s2;
        Vector &p = s3;

        double curv = alignAxis();

        //    auto eff_grad = [&]() { return -dot(g_0, N) * N; };
        auto eff_grad = [&]() { return effGrad(); };

        gf_n = eff_grad();
        p = -gf_n;

        double m = S_MIN;

        for (int i = 0; i < IE_MAX; ++i) {
            if (curv < 0 || dot(g_0, g_0) < F_TOL * F_TOL) {
                return true;
            }

            translate(m * p / std::sqrt(dot(p, p)));

            curv = alignAxis();

            using std::swap;
            swap(gf_n, gf_p);

            gf_n = eff_grad();

            double b = dot(gf_n, gf_n - gf_p) / dot(gf_p, gf_p);

            Vector prev = p;

            p = b * p - gf_n;

            double ps = dot(prev, p);

            if (ps > 0 * dot(p, p)) {
                m = std::min(S_MAX, 1.5 * m);
            } else {
                m = S_MIN;
            }
        }

        return false;
    }

    bool findSaddle() {
        if (!escapeConvex()) {
            return false;
        }

        Vector &p = s1;
        Vector &g_eff = s2;

        alignAxis();

        for (int i = 0; i < IT_MAX; ++i) {
            // reversing perpendicular component does not change magnitude
            if (dot(g_0, g_0) < F_TOL * F_TOL) {
                return true;
            }

            g_eff = effGrad();

            lbfgs_trn(R_0, g_eff, p); // grad is neg of f

            double norm = std::sqrt(dot(p, p));
            double alpha = norm > S_MAX ? S_MAX / norm : 1.0;
            double phi_0 = dot(g_eff, p);

            translate(alpha * p);

            alignAxis();

            double phi_a = dot(effGrad(), p);

            // optional
            if (phi_a <= phi_0) {
                std::terminate();
                return false;
            }
        }

        return false;
    }
};
