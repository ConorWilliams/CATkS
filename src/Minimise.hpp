#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "Eigen/Core"
#include "L_BFGS.hpp"
#include "utils.hpp"

template <typename F1, typename F2> class Minimise {
    static constexpr double C1 = 1e-4;
    static constexpr double C2 = 0.9;

    static constexpr int I_MAX = 1000;
    static constexpr double F_TOL = 1e-6;

    static constexpr double S_MAX = 1;

    F1 const &f;
    F2 const &grad;

    CoreLBFGS<8> lbfgs;

    Vector g;
    Vector p;

    // scratch space
    Vector x0;

  public:
    Minimise(F1 const &f, F2 const &grad, long dims)
        : f{f}, grad{grad}, lbfgs(dims), g{dims}, p{dims}, x0{dims} {}

    bool findMin(Vector &x) {

        lbfgs.clear();

        grad(x, g);

        for (int i = 0; i < I_MAX; ++i) {

            if (dot(g, g) < F_TOL * F_TOL) {
                return true;
            }

            lbfgs(x, g, p);

            double a = 1;

            x0 = x;

            double const f0 = f(x);
            double const g0 = dot(g, p);

            // force descent direction
            if (g0 > 0) {
                p = -p;
            }

            for (int i = 1;; ++i) {
                x = x0 + a * p;
                grad(x, g);

                double fa = f(x);

                if (fa <= f0 + C1 * a * g0) {
                    break;
                } else {
                    double quad = -a * a * g0 * 0.5 / (fa - a * g0 - f0);
                    if (quad <= 0 || quad >= a) {
                        a = a / 2;
                    } else {
                        a = quad;
                    }
                }

                if (i > 20) {
                    std::cerr << "fail in minimiser line search" << std::endl;
                    std::terminate();
                }
            }

            std::cout << x(0) << ' ' << x(1) << ' ' << 1 << ' ' << 0
                      << std::endl;
        }

        std::cerr << "line search failed to converge" << std::endl;
        return false;
    }
};
