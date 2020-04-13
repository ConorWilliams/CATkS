
#include <iostream>

#include "L_BFGS.hpp"

int main() {

    constexpr int N = 8;

    CoreLBFGS<8> core(N);

    Eigen::ArrayXd in(N);

    for (int i = 0; i < N; ++i) {
        in(i) = i;
    }

    std::cout << core.step(in, in) << std::endl;

    core.dump();

    std::cout << "working" << std::endl;

    return 0;
}
