#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>

#include "pcg_random.hpp"

#include "DumpXYX.hpp"
#include "Eigen/Core"
#include "L_BFGS.hpp"
#include "Minimise.hpp"
#include "utils.hpp"

template <typename F, typename P> class Dimer {
    static constexpr int IR_MAX = 15;
    static constexpr int IE_MAX = 50;
    static constexpr int IT_MAX = 200;

    static constexpr double DELTA_R = 0.01;

    static constexpr double /*    */ F_TOL = 1e-5;
    static constexpr double /**/ THETA_TOL = 1 * (2 * M_PI / 360); // deg
    static constexpr double /*    */ G_TOL = 0.01;
    static constexpr double /* */ CURV_TOL = -10;

    static constexpr double S_MIN = 0.1;
    static constexpr double S_MAX = 0.5;

    // increase for more component parallel to dimer, default = 1
    // higher = less relaxation
    static constexpr double BOOST = 0.5;

    F const &grad;
    P printer;
    long dims;

    CoreLBFGS<4> lbfgs_rot;
    CoreLBFGS<16> lbfgs_trn;

    Vector const &init;
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

  private:
    inline void updateGrad() { grad(R_0, g_0); }

    template <typename T> inline void translate(T const &x) {
        printer();
        R_0 += x;
        updateGrad();
    }

    // Effective gradient is negative of translational force acting on dimer
    inline auto effGrad() const { return g_0 - (1 + BOOST) * dot(g_0, N) * N; }

  public:
    // grad is a function-like object such that grad(x, g) writes the gradient
    // at x into g: Vector, Vector -> void
    // R_in is the center of the dimer.
    // N_in is the dimer axis unit vector.
    Dimer(F const &grad, Vector const &init, Vector &R_in, Vector &N_in,
          P const &printer)
        : grad{grad}, printer{printer}, dims{R_in.size()}, lbfgs_rot(dims),
          lbfgs_trn(dims), init{init}, R_0{R_in}, N{N_in}, g_0{dims}, s1{dims},
          s2{dims}, s3{dims}, g_1{dims}, gp_1{dims}, Np{dims}, theta{dims} {}

  private:
    // Aligns the dimer with the minimum curvature mode, returns the curvature
    // along the minimum mode.
    inline double alignAxis() {
        lbfgs_rot.clear();
        grad(R_0 + DELTA_R * N, g_1); // test unevaluated R in grad slow-down

        for (int iter = 0;; ++iter) {
            g_delta = g_1 - g_0;
            lbfgs_rot(N, g_delta, theta); // could use perpendicularised:
                                          // g_delta -= dot(g_delta, N) * N;
            theta -= dot(theta, N) * N;
            theta *= 1 / std::sqrt(dot(theta, theta));

            double b_1 = dot(g_delta, theta) / DELTA_R;
            double c_x0 = dot(g_delta, N) / DELTA_R;
            double theta_1 = -0.5 * std::atan(b_1 / abs(c_x0));

            // std::cout << iter << " theta " << theta_1 << std::endl;

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

                // std::cout << iter << " theta " << theta_min << std::endl;

                N = N * std::cos(theta_min) + theta * std::sin(theta_min);

                g_1 =
                    (std::sin(theta_1 - theta_min) / std::sin(theta_1)) * g_1 +
                    (std::sin(theta_min) / std::sin(theta_1)) * gp_1 +
                    (1 - std::cos(theta_min) -
                     std::sin(theta_min) * std::tan(0.5 * theta_1)) *
                        g_0;

                if (abs(theta_min) < THETA_TOL) {
                    return dot(g_1 - g_0, N) / DELTA_R;
                }
            }
        }
    }

    // Move the dimer out of the convex region into a region were the minimum
    // curvature mode has negative curvature.
    bool escapeConvex() {

        updateGrad();

        Vector const &p = s1 = (R_0 - init).matrix().normalized();

        for (std::size_t iter = 1; iter <= IE_MAX; ++iter) {

            if (alignAxis() < CURV_TOL) {
                // std::cout << iter << std::endl;
                return true;
            }

            double mag = std::min(S_MIN * iter, S_MAX);

            translate(mag * p);
        }

        return false;
    }

  public:
    // Moves the dimer to a saddle point and aligns the dimer axis with the
    // saddle points minimum mode.
    bool findSaddle() {
        lbfgs_trn.clear();

        if (!escapeConvex()) {
            std::cerr << "failed to escape convex" << std::endl;
            return false;
        }
        // { // else
        //     updateGrad();
        //     alignAxis();
        // }

        // std::cerr << "escaped convex" << std::endl;

        Vector &p = s1;
        Vector &g_eff = s2;

        g_eff = effGrad();

        double trust = S_MIN; // S_MIN <~ trust <~ S_MAX
        double g0 = HUGE_VAL; // +infinity

        double curv = alignAxis();
        int strikes = 0;

        for (int iter = 0; iter < IT_MAX; ++iter) {
            // std::cout << "G_eff^2: " << dot(g_0, g_0) << ' ' << curv
            //           << std::endl;

            if (dot(g_eff, g_eff) < F_TOL * F_TOL) {
                return true;
            }

            lbfgs_trn(R_0, g_eff, p);

            translate(std::min(1.0, trust / std::sqrt(dot(p, p))) * p);

            curv = alignAxis();
            g_eff = effGrad();
            double ga = dot(g_eff, p); // -p0

            if (ga >= G_TOL && trust >= S_MIN) {
                trust /= 2;
            } else if (ga <= -G_TOL && trust <= S_MAX) {
                trust *= 2;
            }

            // // optional // //
            if (ga <= g0 && curv > 0) {
                // std::cerr << "STRIKE: " << strikes + 1 << std::endl;
                if (++strikes == 5) {
                    return false;
                }
                // return findSaddle();
                // lbfgs_trn.clear();
                // return false;
            }

            using std::swap;
            swap(g0, ga);
        }

        // std::cerr << "failed converge dimer" << std::endl;
        return false;
    }
};

// controls displacement along mm at saddle
inline constexpr double NUDGE = 0.05;
// tollerence for 3N vectors to be considered the same vector
inline constexpr double TOL_NEAR = 0.1;

constexpr double G_SPHERE = 4;
constexpr double G_AMP = 0.1; // 0.325 pure Fe

// init is a minimised (unporturbed) vector of atoms
// idx is centre of displacemnet
// f is their force object
template <typename F, typename MinImage>
std::vector<std::tuple<Vector, Vector>>
findSaddle(std::size_t attempts, Vector const &init, std::size_t idx,
           F const &f, MinImage const &mi) {

    CHECK(attempts > 0, "trying nothing");

    // Seed with a real random value, if available
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::normal_distribution<> gauss_dist(0, G_AMP);

    Vector sp{init.size()};
    Vector ax{init.size()};

    Dimer dimer{
        f, init, sp, ax, [&]() { /*output(sp, f.quasiColourAll(sp)); */ },
    };

    Minimise min{f, f, init.size()};

    Vector old;
    Vector end;

    std::vector<std::tuple<Vector, Vector>> results;

    for (std::size_t anon = 0; anon < attempts; ++anon) {

        for (int i = 0; i < init.size(); i = i + 3) {

            Eigen::Vector3d delta{
                init[idx * 3 + 0] - init[i + 0],
                init[idx * 3 + 1] - init[i + 1],
                init[idx * 3 + 2] - init[i + 2],
            };

            double dist = mi(delta).squaredNorm();

            if (dist < G_SPHERE * G_SPHERE) {
                sp[i + 0] = init[i + 0] + gauss_dist(rng);
                sp[i + 1] = init[i + 1] + gauss_dist(rng);
                sp[i + 2] = init[i + 2] + gauss_dist(rng);

                ax[i + 0] = gauss_dist(rng);
                ax[i + 1] = gauss_dist(rng);
                ax[i + 2] = gauss_dist(rng);
            } else {
                sp[i + 0] = init[i + 0];
                sp[i + 1] = init[i + 1];
                sp[i + 2] = init[i + 2];

                ax[i + 0] = 0;
                ax[i + 1] = 0;
                ax[i + 2] = 0;
            }
        }

        ax.matrix().normalize();

        if (!dimer.findSaddle()) {
            // std::cout << "fail sp search\n";
            // failed SP search
            continue;
        }

        old = sp + ax * NUDGE;
        end = sp - ax * NUDGE;

        if (!min.findMin(old) || !min.findMin(end)) {
            // failed minimisation
            // std::cout << "fail minim search\n";
            continue;
        }

        double distOld = dot(old - init, old - init);
        double distFwd = dot(end - init, end - init);

        // want old to be init
        if (distOld > distFwd) {
            using std::swap;
            swap(old, end);
            swap(distOld, distFwd);
        }

        if (distOld > TOL_NEAR) {
            // std::cout << "disconeced sp \n";
            // disconnected SP
            continue;
        }

        if (dot(end - old, end - old) < TOL_NEAR) {
            // std::cout << "loop de loop \n";
            // minimasations both converged to init
            continue;
        }

        results.emplace_back(sp, end);
    }

    return results;
}
