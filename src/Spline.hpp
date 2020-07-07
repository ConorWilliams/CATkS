#pragma once

#include <vector>

#include "utils.hpp"

class NaturalSpline {
  private:
    struct Spline {
        double a, b, c, d;
    };

    std::vector<Spline> splines;

    double dx;
    double idx;

    std::size_t n;

  public:
    NaturalSpline(double dx) : dx{dx}, idx{1 / dx} {};

    inline double operator()(double x) {
        CHECK(x >= 0, "out of bounds " << x);

        const std::size_t i = x * idx;

        CHECK(i < n, "out of bounds " << x << ' ' << n);

        x -= i * dx;

        return splines[i].a +
               x * (splines[i].b + x * (splines[i].c + x * splines[i].d));
    }

    inline double grad(double x) {
        CHECK(x >= 0, "out of bounds " << x);

        const std::size_t i = x * idx;

        CHECK(i < n, "out of bounds " << x << ' ' << n);

        x -= i * dx;

        return splines[i].b + x * (2 * splines[i].c + 3 * x * splines[i].d);
    }

    // y is (n + 1) y_i values evenly spaced on interval 0,dx,...,ndx
    template <typename T> void computeCoeff(T const &y) {
        n = y.size() - 1;
        // 1
        std::vector<double> a(n + 1);

        for (std::size_t i = 0; i <= n; ++i) {
            a[i] = y[i];
        }

        // 2
        std::vector<double> b(n);
        std::vector<double> d(n);

        // 4
        std::vector<double> alpha(n);

        for (std::size_t i = 1; i <= n - 1; ++i) {
            const double inv_h = 3 / dx;

            alpha[i] = inv_h * (a[i + 1] - 2 * a[i] + a[i - 1]);
        }

        // 5
        std::vector<double> c(n + 1);
        std::vector<double> l(n + 1);
        std::vector<double> mu(n + 1);
        std::vector<double> z(n + 1);

        // 6
        l[0] = 1;
        mu[0] = 0;
        z[0] = 0;

        // 7
        for (std::size_t i = 1; i <= n - 1; ++i) {
            l[i] = dx * (4 - mu[i - 1]);
            mu[i] = dx / l[i];
            z[i] = (alpha[i] - dx * z[i - 1]) / l[i];
        }

        // 8
        l[n] = 1;
        z[n] = 0;
        c[n] = 0;

        // 9
        for (std::size_t i = 1; i <= n; ++i) {
            std::size_t j = n - i;

            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / dx - dx * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * dx);
        }

        // 11
        splines.clear();

        for (std::size_t i = 0; i <= n - 1; ++i) {
            splines.push_back({a[i], b[i], c[i], d[i]});
        }
    }
};
