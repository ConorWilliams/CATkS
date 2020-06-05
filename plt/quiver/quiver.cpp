
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "2d_pot.hpp"
#include "Eigen/Core"
#include "L_BFGS.hpp"
#include "utils.hpp"

template <typename F> class Dimer {
    static constexpr int IR_MAX = 50;
    static constexpr int IE_MAX = 50;
    static constexpr int IT_MAX = 1950;

    static constexpr double DELTA_R = 0.001;

    static constexpr double /*    */ F_TOL = 1e-8;
    static constexpr double /**/ THETA_TOL = 0.005 * (2 * M_PI / 360); // 1deg
    static constexpr double /*    */ G_TOL = 0.01;

    static constexpr double S_MIN = 0.1;
    static constexpr double S_MAX = 0.5;

    F const &grad;
    long dims;

    CoreLBFGS<4> lbfgs_rot;
    CoreLBFGS<8> lbfgs_trn;

    Vector &R_0;
    Vector &N;

    Vector g_0;

    // shared space
    Vector s1;
    Vector s2;
    Vector s3;
    Vector s4;

    // rotation storage
    Vector g_1;
    Vector gp_1;
    Vector g_delta;
    Vector Np;
    Vector theta;

  public:
    inline void updateGrad() { grad(R_0, g_0); }

    template <typename T> inline void translate(T const &x) {
        print();
        R_0 += x;
        updateGrad();
    }

    // Effective gradient is negative of translational force acting on dimer
    inline auto effGrad() const { return g_0 - 2 * dot(g_0, N) * N; }

    inline auto effGradPar() const { return dot(g_0, N) * N; }
    inline auto effGradPerp() const { return g_0 - dot(g_0, N) * N; }
    inline auto actualGrad() const { return N; }

    void print() const {
        // std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
        //           << ' ' << R_0(2) << ' ' << R_0(3) << std::endl;
        std::cout << R_0(0) << ' ' << R_0(1) << ' ' << N(0) << ' ' << N(1)
                  << std::endl;
    }

    // grad is a function-like object such that grad(x, g) writes the gradient
    // at x into g: Vector, Vector -> void
    // R_in is the center of the dimer.
    // N_in is the dimer axis unit vector.
    Dimer(F const &grad, Vector &R_in, Vector &N_in)
        : grad{grad}, dims{R_in.size()}, lbfgs_rot(dims),
          lbfgs_trn(dims), R_0{R_in}, N{N_in}, g_0{dims}, s1{dims}, s2{dims},
          s3{dims}, g_1{dims}, gp_1{dims}, Np{dims}, theta{dims} {
        N /= std::sqrt(dot(N, N));
    }

    // Aligns the dimer with the minimum curvature mode, returns the curvature
    // along the minimum mode.
    inline double alignAxis() {
        lbfgs_rot.clear();
        grad(R_0 + DELTA_R * N, g_1); // test unevaluated R in grad slow-down

        for (int iter = 0;; ++iter) {
            g_delta = g_1 - g_0;
            lbfgs_rot(N, g_delta, theta); // could use perpendicularised

            theta -= dot(theta, N) * N;
            theta /= std::sqrt(dot(theta, theta));

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
};

struct Grad {
    inline void operator()(Vector const &pos, Vector &result) const {
        auto x1 = pos(0);
        auto y1 = pos(1);

        result(0) = potentials::fpx(x1, y1);

        result(1) = potentials::fpy(x1, y1);
    }
};

struct Pot {
    inline double operator()(Vector const &pos) const {
        auto x = pos(0);
        auto y = pos(1);

        return potentials::f(x, y);
    }
};

int main() {
    constexpr int N = 2;
    Vector pos{N};
    Vector axis{N};

    pos(0) = 0.77;
    pos(1) = -0.07;

    axis(0) = 1;
    axis(1) = 1;

    axis.matrix().normalize();

    Dimer dimer{Grad{}, pos, axis};

    // xrange = [0.5, 3]
    // yrange = [-3, 3.3]
    double delta_x = (3 - 0.67) / 25;
    double delta_y = (3.3 - -3) / 25;

    for (double x = 0.67; x <= 3; x = x + delta_x) {
        for (double y = -3; y <= 3.3; y = y + delta_y) {
            pos(0) = x;
            pos(1) = y;

            axis(0) = 1;
            axis(1) = 1;

            dimer.updateGrad();
            dimer.alignAxis();

            axis.matrix().normalize();

            // Vector force = -(dimer.effGradPerp() - 2 * dimer.effGradPar());
            Vector force = -dimer.actualGrad();
            std::cout << pos(0) << ' ' << pos(1) << ' ' << force(0) << ' '
                      << force(1) << std::endl;
        }
    }

    std::cerr << "working" << std::endl;
}
