#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "Eigen/Core"
#include "L_BFGS.hpp"
#include "utils.hpp"

#include <iomanip>

template <typename F1, typename F2> class Minimise {
    static constexpr double C1 = 1e-4;
    static constexpr double C2 = 0.9;

    static constexpr int I_MAX = 1000;
    static constexpr int L_MAX = 3;

    static constexpr double F_TOL = 1e-5;

    static constexpr double S_MAX = 1;

    F1 const &f;
    F2 const &grad;

    CoreLBFGS<8> lbfgs;

    Vector g;
    Vector p;

    // scratch space
    Vector x0;

  public:
    // f is a function-like object such that f(x) return the objective function
    // (potential energy) at x : Vector -> double
    // grad is a function-like object such that grad(x, g) writes the gradient
    // at x into g: Vector, Vector -> void
    // dims is the number of dimensions in the problem minimisation space e.g
    // the length of the Vectors g and x.
    Minimise(F1 const &f, F2 const &grad, long dims)
        : f{f}, grad{grad}, lbfgs(dims), g{dims}, p{dims}, x0{dims} {}

    // Moves x to a local minimum;
    bool findMin(Vector &x) {
        lbfgs.clear();

        grad(x, g);

        for (int iter = 0; iter < I_MAX; ++iter) {
            // std::cout << "Min iter: " << iter << ' ' << std::setprecision(16)
            //           << f(x) << std::endl;

            if (dot(g, g) < F_TOL * F_TOL) {
                return true;
            }

            x0 = x; // pre move location saved

            lbfgs(x, g, p);

            double a = 1;

            //    double const f0 = f(x);
            double const g0 = dot(g, p);

            // force descent direction
            if (g0 > 0) {
                p = -p;
            }

            // for (int l2 = 0; l2 < 9; ++l2) {
            //     std::cout << p[l2] << ' ';
            // }
            // std::cout << std::endl;

            // dumb line search;
            double norm = std::sqrt(dot(p, p));
            a = norm > 0.5 ? 0.5 / norm : 1;
            x = x0 + a * p;
            grad(x, g);

            // backtracking line search
            // for (int i = 1;; ++i) {
            //     x = x0 + a * p;
            //     grad(x, g);
            //
            //     double fa = f(x);
            //
            //     // Wolfie sufficiant decrese condition
            //     if (fa <= f0 + C1 * a * g0) {
            //         break;
            //     } else {
            //         double quad = a * a * g0 * 0.5 / (a * g0 - fa + f0);
            //         if (quad <= 0 || quad >= a) {
            //             a = a / 2;
            //         } else {
            //             a = quad;
            //         }
            //     }
            //
            //     if (i > L_MAX) {
            //         std::cerr << "fail in minimiser line search" <<
            //         std::endl;
            //         // break;
            //         return false;
            //     }
            // }

            // std::cout << x(0) << ' ' << x(1) << ' ' << 1 << ' ' << 0
            //           << std::endl;
        }

        std::cerr << "line search failed to converge" << std::endl;
        return false;
    }
};
