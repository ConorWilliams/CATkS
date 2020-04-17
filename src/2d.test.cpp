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
        auto x1 = pos(0);
        auto y1 = pos(1);

        result(0) = potentials::fpx(x1, y1);

        result(1) = potentials::fpy(x1, y1);

        // auto x2 = pos(2);
        // auto y2 = pos(3);
        //
        // result(2) = potentials::fpx(x2, y2);
        //
        // result(3) = potentials::fpy(x2, y2);
    }
};

#include <random>

int main() {
    Vector pos{2};
    Vector axis{2};

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<double> d{0, 0.2};
    std::uniform_real_distribution<double> u(-M_PI, M_PI);

    pos(0) = 0.77 + d(gen);
    pos(1) = -0.07 + d(gen);

    // pos(2) = 0.77 + d(gen);
    // pos(3) = -0.07 + d(gen);

    double theta = u(gen);

    axis(0) = std::cos(theta);
    axis(1) = std::sin(theta);

    // axis(2) = std::sin(theta);
    // axis(3) = std::cos(theta);

    axis.matrix().normalize();

    dimerSearch(Grad4{}, pos, axis);

    std::cout << pos(0) << ' ' << pos(1) << ' ' << axis(0) << ' ' << axis(1)
              << std::endl;

    // std::cout << pos(0) << ' ' << pos(1) << ' ' << axis(0) << ' ' << axis(1)
    //           << ' ' << pos(2) << ' ' << pos(3) << std::endl;

    return 0;
}
