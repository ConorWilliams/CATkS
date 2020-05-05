#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "sort.hpp"

decltype(std::chrono::system_clock::now()) START;

inline void tick() { START = std::chrono::high_resolution_clock::now(); }

inline int tock() {
    auto const stop = std::chrono::high_resolution_clock::now();
    auto const time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - START)
            .count();

    return time;
}

int main() {

    std::random_device rd{};
    std::mt19937 gen{rd()};

    for (int i = 4; i < 28; ++i) {
        std::vector<int> averages(6, 0);
        constexpr int min = 0;
        int max = std::pow(10.0, (double)i / 4);

        std::uniform_int_distribution<int> u(min, max);
        std::uniform_int_distribution<int> u2(min, max - 1);

        auto length = 30 * max;

        int repeat = 28 - i;

        for (int j = 0; j < repeat; ++j) {
            std::cerr << length << ' ' << j << std::endl;

            std::vector<int> ints1(length);

            for (auto &&elem : ints1) {
                elem = u(gen);
            }
            std::vector<int> ints2 = ints1;
            std::vector<int> ints3 = ints1;

            ////////////////////////////////////////////////////

            tick();
            std::sort(ints1.begin(), ints1.end());
            averages[0] += tock();

            for (std::size_t i = 0; i < ints1.size() / 100; ++i) {
                ints1[u2(gen)] = u(gen);
            }

            tick();
            std::sort(ints1.begin(), ints1.end());
            averages[1] += tock();

            //////////////////////////////////////////////////

            tick();
            cj::sort(ints2.begin(), ints2.end(), min, max);
            averages[2] += tock();

            for (std::size_t i = 0; i < ints2.size() / 100; ++i) {
                ints2[u2(gen)] = u(gen);
            }

            tick();
            cj::sort(ints2.begin(), ints2.end(), min, max);
            averages[3] += tock();

            //////////////////////////////////////////////////

            tick();
            ints3 = cj::sort2(ints3, min, max);
            averages[4] += tock();

            for (std::size_t i = 0; i < ints3.size() / 100; ++i) {
                ints3[u2(gen)] = u(gen);
            }

            tick();
            ints3 = cj::sort2(ints3, min, max);
            averages[5] += tock();
        }

        std::cout << length;
        for (auto &&e : averages) {
            std::cout << ' ' << (double)e / repeat;
        }
        std::cout << std::endl;
    }

    return 0;
}
