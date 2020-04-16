#include <iostream>

#include "Eigen/Core"

#include "2d_pot.hpp"
#include "dimer.hpp"
#include "utils.hpp"

// f= 7xy e ^ (-x2+-y2)
struct Grad3 {
    inline void operator()(Vector const &pos, Vector &result) {
        auto x = pos(0);
        auto y = pos(1);

        auto f = 7 * std::exp(-x * x - y * y);

        result(0) = y * f * (1 - 2 * x * x);

        result(1) = x * f * (1 - 2 * y * y);
    }
};

struct Grad4 {
    inline void operator()(Vector const &pos, Vector &result) {
        auto x = pos(0);
        auto y = pos(1);

        result(0) = potentials::fpx(x, y);

        result(1) = potentials::fpy(x, y);
    }
};

int main() {
    Vector pos{2};
    Vector axis{2};

    pos(0) = 1;
    pos(1) = -0.5;

    axis(0) = 1.2;
    axis(1) = 1;

    axis.matrix().normalize();

    dimerSearch(Grad4{}, pos, axis);

    std::cout << pos(0) << ' ' << pos(1) << ' ' << axis(0) << ' ' << axis(1)
              << std::endl;

    return 0;
}
