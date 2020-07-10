
#include <iostream>

#include "L_BFGS.hpp"
#include <cmath>

auto grad(Eigen::ArrayXd const &pos) {
    auto x = pos(0);
    auto y = pos(1);

    auto x2 = 4 * x * (x * x + y - 11) + 2 * (x + y * y - 7);

    auto y2 = 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7);

    return Eigen::Array<double, 2, 1>{x2, y2};
}

auto f(Eigen::ArrayXd const &pos) {
    auto x = pos(0);
    auto y = pos(1);

    return std::pow(x * x + y - 11, 2) + std::pow(x + y * y - 7, 2);
}

auto f2(Eigen::ArrayXd const &pos) { return grad(pos).matrix().squaredNorm(); }

int main() {

    constexpr int dims = 2;

    Eigen::ArrayXd pos(dims);

    CoreLBFGS<10> core(pos.size());

    pos(0) = 3.25;
    pos(1) = -0.9;

    std::cout << std::endl;

    int count = 0;

    Eigen::ArrayXd step;

    while (f2(pos) > 0.001) {
        count++;

        core(pos, grad(pos), step);

        // while (f2(pos - step) > f2(pos) + 0.001 * (step * grad(pos)).sum()) {
        //     std::cout << "yes" << std::endl;
        //     step /= 1.5;
        // }

        auto norm = std::sqrt((step * step).sum());
        double max = 1;
        if (norm > max) {
            std::cout << "yes" << std::endl;
            step /= norm * max;
        }

        // std::cout << "step: " << step.transpose() << std::endl;

        pos -= step;

        std::cout << count << ": " << pos.transpose() << ' ' << f(pos)
                  << std::endl;
    }

    std::cout << "working" << std::endl;

    return 0;
}
