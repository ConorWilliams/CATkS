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

int main() {
    Vector pos{2};
    Vector axis{2};

    pos(0) = 10;
    pos(1) = 10;

    axis(0) = 1;
    axis(1) = 1;

    axis.matrix().normalize();

    dimerSearch<Grad>(pos, axis);

    pp(pos);
    pp(axis);

    std::cout << "working" << std::endl;

    return 0;
}