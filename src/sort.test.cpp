#include <algorithm>
#include <iostream>

#include "Eigen/Core"
#include "sort.hpp"
#include "time.hpp"
#include "utils.hpp"

#include <random>

int main() {

    std::random_device rd{};
    std::mt19937 gen{rd()};

    constexpr int min = 0;
    constexpr int max = 1000000 / 25;

    std::uniform_int_distribution<int> u(min, max);

    Eigen::Array<int, Eigen::Dynamic, 1> ints{25 * max};

    for (auto &&elem : ints) {
        elem = u(gen);
    }

    Eigen::Array<int, Eigen::Dynamic, 1> ints_cpy = ints;

    tick();
    std::sort(ints_cpy.begin(), ints_cpy.end());
    tock("  std::sort");

    tick();
    std::sort(ints_cpy.begin(), ints_cpy.end());
    tock("  std::sort");

    tick();
    sort(ints.begin(), ints.end(), min, max);
    tock("direct_sort");

    tick();
    sort(ints.begin(), ints.end(), min, max, [](auto x) { return x; });
    tock("direct_sort");

    assert((ints_cpy == ints).all());

    std::cout << "Done" << std::endl;
    return 0;
}
