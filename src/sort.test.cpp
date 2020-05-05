#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "sort.hpp"

using clk = std::chrono::high_resolution_clock;

decltype(clk::now()) START;

inline void tick() { START = clk::now(); }

inline double tock() {
    auto stop = clk::now();
    return std::chrono::duration<double, std::ratio<1, 1000>>(stop - START)
        .count();
}

#include <array>

struct vec3 {
    double a;
    double b;
    double c;
};

constexpr int min = 0;

int main() {

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // 2-24
    for (int i = 2; i <= 30; ++i) {
        std::vector<double> averages(6, 0);

        int length = std::pow(10.0, (double)i / 4);

        int max = std::max(length / 25, 10);

        std::uniform_real_distribution<double> u(min, max);
        std::uniform_int_distribution<int> u2(min, max - 1);

        int repeat = 31 - i;

        for (int j = 0; j < repeat; ++j) {

            std::vector<vec3> ints1(length);

            for (auto &&elem : ints1) {
                elem.a = u(gen);
                elem.b = u(gen);
                elem.c = u(gen);
            }
            std::vector<vec3> ints2 = ints1;
            std::vector<vec3> ints3 = ints1;

            const auto lam = [](vec3 const &x) {
                return ((int)x.a + (int)x.b + (int)x.c) / 3;
            };

            ////////////////////////std////////////////////////////

            tick();
            std::sort(ints1.begin(), ints1.end(),
                      [&](vec3 x, vec3 y) { return lam(x) < lam(y); });
            averages[0] += tock();

            for (std::size_t i = 0; i < ints1.size() / 100; ++i) {
                ints1[u2(gen)].a = u(gen);
            }

            tick();
            std::sort(ints1.begin(), ints1.end(),
                      [&](vec3 x, vec3 y) { return lam(x) < lam(y); });
            averages[1] += tock();

            ////////////////////////swap_count////////////////////////

            tick();
            cj::sort(ints2.begin(), ints2.end(), min, max, lam);
            averages[2] += tock();

            for (std::size_t i = 0; i < ints2.size() / 100; ++i) {
                ints2[u2(gen)].a = u(gen);
            }

            tick();
            cj::sort(ints2.begin(), ints2.end(), min, max, lam);
            averages[3] += tock();

            ////////////////////////////////////////////////

            // for (auto &&elem : ints3) {
            //     std::cout << lam(elem) << ' ';
            // }
            // std::cout << std::endl;

            tick();
            cj::sort_clever(ints3.begin(), ints3.end(), min, max, lam);
            averages[4] += tock();

            // for (auto &&elem : ints3) {
            //     std::cout << lam(elem) << ' ';
            // }
            // std::cout << std::endl;

            for (std::size_t i = 0; i < ints3.size() / 100; ++i) {
                ints3[u2(gen)].a = u(gen);
            }

            tick();
            cj::sort_clever(ints3.begin(), ints3.end(), min, max, lam);
            averages[5] += tock();

            ////////////////////////////////////////////////
        }

        std::cout << length;
        for (auto &&e : averages) {
            std::cout << ' ' << e / repeat;
        }
        std::cout << std::endl;
    }

    return 0;
}
