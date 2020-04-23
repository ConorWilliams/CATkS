#include <iostream>

#include "Eigen/Core"

#include "2d_pot.hpp"
#include "Dimer.hpp"
#include "Minimise.hpp"
#include "utils.hpp"

// f= 7xy e ^ (-x2+-y2)

struct Grad {
    inline void operator()(Vector const &pos, Vector &result) const {
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

struct Pot {
    inline double operator()(Vector const &pos) const {
        auto x = pos(0);
        auto y = pos(1);

        return potentials::f(x, y);
    }
};

#include <random>

int main() {
    constexpr int N = 2;
    Vector pos{N};
    Vector axis{N};

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<double> d{0, 0.1};
    std::uniform_real_distribution<double> u(-M_PI, M_PI);

    pos(0) = 0.77 - 0.05;
    pos(1) = -0.07 - 0.4;
    // pos(2) = 0.692;
    // pos(3) = 3.801;

    [[maybe_unused]] double theta = u(gen);

    axis(0) = std::cos(theta);
    axis(1) = std::sin(theta);

    // axis(2) = std::sin(theta);
    // axis(3) = std::cos(theta);

    axis.matrix().normalize();

    auto rand = Eigen::ArrayXXd::NullaryExpr(N, 1, [&]() { return d(gen); });

    pos += rand;

    Dimer dimer{Grad{}, pos, axis};

    if (!dimer.findSaddle()) {
        std::cerr << "saddle fail" << std::endl;
        return 0;
    }
    //
    // Minimise min{Pot{}, Grad{}, pos.size()};
    //
    // pos += 0.1 * axis;
    // // dimer.print();
    //
    // min.findMin(pos);

    // std::cout << pos(0) << ' ' << pos(1) << ' ' << axis(0) << ' ' <<
    // axis(1)
    //           << ' ' << pos(2) << ' ' << pos(3) << std::endl;

    return 0;
}
