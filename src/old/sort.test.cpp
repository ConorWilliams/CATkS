#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "sort.hpp"

// Helper class holds atom in LIST array of LCL with index of neighbours
template <typename T> class Atom {
  private:
    std::array<double, 3> r;
    double m_rho;
    T k;
    std::size_t n;

  public:
    Atom(T k, double x, double y, double z) : r{x, y, z}, m_rho{0.0}, k{k} {}
    Atom(Atom const &) = default;
    Atom(Atom &&) = default;
    Atom() = default;
    Atom &operator=(Atom const &) = default;
    Atom &operator=(Atom &&) = default;

    inline T const &kind() const { return k; }

    inline std::size_t &next() { return n; }
    inline std::size_t next() const { return n; }

    inline double &rho() { return m_rho; }
    inline double rho() const { return m_rho; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

constexpr int MIN = 0;

size_t num_atoms = 100;
double L = std::cbrt(num_atoms / 2) * 2.87;
std::size_t M = static_cast<std::size_t>((L) / 6) + 2;
std::size_t MAX = M * M * M;

template <typename T> inline std::size_t lam(Atom<T> const &atom) {

    std::size_t i = (atom[0] - 0) * M / L;
    std::size_t j = (atom[1] - 0) * M / L;
    std::size_t k = (atom[2] - 0) * M / L;

    return i + j * M + k * M * M;
}

using clk = std::chrono::high_resolution_clock;

decltype(clk::now()) START;

inline void tick() { START = clk::now(); }

inline double tock() {
    auto stop = clk::now();
    return std::chrono::duration<double, std::ratio<1, 1000000000>>(stop -
                                                                    START)
        .count();
}

int main() {
    std::cout << sizeof(Atom<int>) << std::endl;

    std::terminate();

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // 2-28
    for (int i = 4; i <= 26; ++i) {
        std::vector<double> averages(6, 0);

        num_atoms = std::pow(10.0, (double)i / 4);
        L = std::cbrt(num_atoms / 2) * 2.87;
        M = static_cast<std::size_t>((L) / 6) + 2;
        MAX = M * M * M;

        std::uniform_real_distribution<double> u(0, L);
        std::uniform_int_distribution<int> u2(0, num_atoms - 1);

        int repeat = 100 - 2 * i;

        for (int j = 0; j < repeat; ++j) {

            std::vector<Atom<int>> ints1(num_atoms);

            for (auto &&elem : ints1) {
                elem[0] = u(gen);
                elem[1] = u(gen);
                elem[2] = u(gen);
            }

            auto ints2 = ints1;
            auto ints3 = ints1;

            ////////////////////////std////////////////////////////

            tick();
            std::sort(ints1.begin(), ints1.end(),
                      [&](auto x, auto y) { return lam(x) < lam(y); });
            averages[0] += tock();

            for (std::size_t i = 0; i < std::max(2UL, ints1.size() / 100);
                 ++i) {
                ints1[u2(gen)][1] = u(gen);
            }

            tick();
            std::sort(ints1.begin(), ints1.end(),
                      [&](auto x, auto y) { return lam(x) < lam(y); });
            averages[1] += tock();

            ////////////////////////swap_count////////////////////////

            tick();
            ints2 = cj::sort2(ints2, MIN, MAX, [](auto x) { return lam(x); });
            averages[2] += tock();

            for (std::size_t i = 0; i < std::max(2UL, ints2.size() / 100);
                 ++i) {
                ints2[u2(gen)][2] = u(gen);
            }

            tick();
            ints2 = cj::sort2(ints2, MIN, MAX, [](auto x) { return lam(x); });
            averages[3] += tock();

            ////////////////////////////////////////////////

            // for (auto &&elem : ints3) {
            //     std::cout << lam(elem) << ' ';
            // }
            // std::cout << std::endl;

            tick();

            cj::sort_clever(ints3.begin(), ints3.end(), MIN, MAX,
                            [](auto x) { return lam(x); });

            averages[4] += tock();

            // for (auto &&elem : ints3) {
            //     std::cout << lam(elem) << ' ';
            // }
            // std::cout << std::endl;

            // check

            for (std::size_t i = 0; i < std::max(2UL, ints3.size() / 100);
                 ++i) {
                ints3[u2(gen)][1] = u(gen);
            }

            tick();
            cj::sort_clever(ints3.begin(), ints3.end(), MIN, MAX,
                            [](auto x) { return lam(x); });
            averages[5] += tock();

            for (auto it = ints3.begin() + 1; it != ints3.end(); ++it) {
                assert(lam(*(it - 1)) <= lam(*it));
            }

            ////////////////////////////////////////////////
        }

        std::cout << num_atoms;
        for (auto &&e : averages) {
            std::cout << ' ' << e / repeat;
        }
        std::cout << std::endl;
    }

    return 0;
}
