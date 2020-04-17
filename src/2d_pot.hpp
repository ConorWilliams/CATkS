#pragma once

#include <cmath>

namespace potentials {

inline constexpr double a = 0.05;
inline constexpr double b = 0.3;
inline constexpr double c = 0.05;
inline constexpr double d1 = 4.746;
inline constexpr double d2 = 4.746;
inline constexpr double d3 = 3.445;
inline constexpr double al = 1.942;
inline constexpr double r0 = 0.742;

inline constexpr double delta = 1e-6;

double Q(double r, double d) {
    return d / 2 *
           (1.5 * std::exp(-2 * al * (r - r0)) - std::exp(-al * (r - r0)));
}

double J(double r, double d) {
    return d / 4 *
           (std::exp(-2 * al * (r - r0)) - 6 * std::exp(-al * (r - r0)));
}

double E(double r1, double r2, double r3) {
    return (Q(r1, d1) / (1 + a) + Q(r2, d2) / (1 + b) + Q(r3, d3) / (1 + c) -
            std::sqrt(std::pow(J(r1, d1) / (1 + a), 2) +
                      std::pow(J(r2, d2) / (1 + b), 2) +
                      std::pow(J(r3, d3) / (1 + c), 2) -
                      J(r1, d1) * J(r2, d2) / ((1 + a) * (1 + b)) -
                      J(r2, d2) * J(r3, d3) / ((1 + b) * (1 + c)) -
                      J(r1, d1) * J(r3, d3) / ((1 + a) * (1 + c))));
}

inline constexpr double rAC = 3.742;
inline constexpr double kc = 0.2025;

double gaus(double A, double x, double y, double x0, double y0, double xi,
            double yi) {
    return A * std::exp(-(x - x0) * (x - x0) / (2 * xi) -
                        (y - y0) * (y - y0) / (2 * yi));
}

double V(double rAB, double rBD) {
    return E(rAB, rAC - rAB, rAC) +
           2 * kc * std::pow(rAB - (rAC / 2 - rBD / 1.154), 2);
}

double f(double x, double y) {
    return V(x, y) + gaus(1.5, x, y, 2.02083, -0.172881, 0.1, 0.35) +
           gaus(6, x, y, 0.8, 2.0, 0.25, 0.7);
}

double fpx(double x, double y) {
    return (f(x + delta, y) - f(x - delta, y)) / (2 * delta);
}

double fpy(double x, double y) {
    return (f(x, y + delta) - f(x, y - delta)) / (2 * delta);
}

} // namespace potentials
