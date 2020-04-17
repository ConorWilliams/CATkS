#include <iostream>

#include "Eigen/Core"

#include "dimer.hpp"
#include "utils.hpp"

struct Grad {
    inline void operator()(Vector const &x, Vector &result) {
        result(0) = 2 * x(0);
        result(1) = -2 * x(1);
    }
};

inline constexpr double A = 1.2;

double gauss(double x, double y) { return std::exp(-(x * x) - (y * y)); }

struct Grad2 {
    inline void operator()(Vector const &pos, Vector &result) {
        auto x = pos(0);
        auto y = pos(1);

        result(0) =
            2 * (x - A) * gauss(x - A, y) + 2 * (x + A) * gauss(x + A, y);

        result(1) = 2 * y * gauss(x - A, y) + 2 * y * gauss(x + A, y);
    }
};

struct Grad3 {
    inline void operator()(Vector const &pos, Vector &result) {
        auto x = pos(0);
        auto y = pos(1);

        auto f = 7 * std::exp(x * x + y * y);

        result(0) = y * f * (1 + 2 * x * x);

        result(1) = x * f * (1 + 2 * y * y);
    }
};

void pp(Vector &v) { std::cout << v.transpose() << std::endl; }

int main() {
    Vector pos{2};
    Vector axis{2};

    pos(0) = -8;
    pos(1) = 34;

    axis(0) = 1;
    axis(1) = 1;

    axis.matrix().normalize();

    std::cout << "first" << std::endl;

    dimerSearch(Grad{}, pos, axis);

    pp(pos);
    pp(axis);

    ///////////////////////////////////////

    pos(0) = A - 0.2;
    pos(1) = 0.3;

    axis(0) = 1;
    axis(1) = 1;

    axis.matrix().normalize();

    std::cout << "second" << std::endl;

    dimerSearch(Grad2{}, pos, axis);

    pp(pos);
    pp(axis);

    ///////////////////////////////////////

    pos(0) = 0.9;
    pos(1) = -0.1;

    axis(0) = 1.2;
    axis(1) = 1;

    axis.matrix().normalize();

    std::cout << "third" << std::endl;

    dimerSearch(Grad3{}, pos, axis);

    pp(pos);
    pp(axis);

    return 0;
}
