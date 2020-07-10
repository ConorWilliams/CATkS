#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "Canon.hpp"

template <typename Atom_t>
inline static bool bonded(Atom_t const &a, Atom_t const &b) {

    // pre H was working at 2.55 // 2.47 -- 2.86 angstrom (just > first neigh)
    static constexpr double F_F_BOND = 2.55;
    static constexpr double H_H_BOND = 2.00; // 0.5^2+0.5^2 vacancy neigh
    static constexpr double F_H_BOND = 2.70; // 0.75^2 + 0.5^2 was 2.7

    static const Eigen::Matrix2d DISTS{
        {F_F_BOND, F_H_BOND},
        {F_H_BOND, H_H_BOND},
    };

    CHECK(a.kind() < 2, "atom type not valid " << a.kind());
    CHECK(b.kind() < 2, "atom type not valid " << b.kind());

    double sqdist = (a.pos() - b.pos()).squaredNorm();

    double bond_len = DISTS(a.kind(), b.kind());

    return sqdist < bond_len * bond_len;
}

class NautyCanon2 {
  private:
    enum : bool { colour = false, plain = true };

    template <typename Atom_t> struct AtomWrap {
        Atom_t *atom;
        double sum;

        AtomWrap(Atom_t &atom) : atom{&atom}, sum{0} {}

        inline Atom_t *operator->() { return atom; }
        inline Atom_t const *operator->() const { return atom; }

        inline Atom_t &operator*() { return *atom; }
        inline Atom_t const &operator*() const { return *atom; }

        inline int toInt() const { return sum; };
    };

  public:
    using Key_t = NautyGraph;

    // NOT thread safe.
    // Writes canonically ordered atoms to order and returns the canonically
    // ordered graph represented as a dense adjecency matrix.
    template <typename Atom_t>
    static inline NautyGraph canonicalize(std::vector<Atom_t> &atoms,
                                          std::vector<Atom_t> &order) {

        CHECK(order.size() == 0, "order has atoms already");

        static int lab[MAXN];
        static int ptn[MAXN];
        static int orbits[MAXN];

        static statsblk stats;

        static DEFAULTOPTIONS_GRAPH(options);

        options.getcanon = true;

        NautyGraph g{};
        NautyGraph cg{};

        std::size_t n = atoms.size();
        std::size_t m = SETWORDSNEEDED(n);

        CHECK(n <= MAXN && m <= MAXM, "too many atoms " << n << ' ' << m);

        nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

        //// order by colour
        options.defaultptn = colour;

        std::vector<AtomWrap<Atom_t>> wrap{atoms.begin(), atoms.end()};

        CHECK(wrap.size() == atoms.size(), "incorrect constuctor deduction");

        for (auto &&a : wrap) {
            for (auto &&b : wrap) {
                a.sum += (a->pos() - b->pos()).squaredNorm();
            }
            CHECK(a.sum < INT_MAX, "overflow gonna getcha");
        }

        std::sort(wrap.begin(), wrap.end(),
                  [](auto const &a, auto const &b) { return a.sum < b.sum; });

        std::cout << "\nHere" << std::endl;
        for (auto &&a : wrap) {
            std::cout << a.sum << ':' << (int)a.sum << '\n';
        }
        // std::terminate();

        for (std::size_t i = 0; i < n; ++i) {
            lab[i] = i;
            if (i + 1 == n || wrap[i + 1].toInt() != wrap[i].toInt()) {
                ptn[i] = 0;
            } else {
                ptn[i] = 1;
            }
        }
        ////

        // Fill in adjecency matrix
        for (std::size_t i = 0; i < wrap.size(); ++i) {
            for (std::size_t j = i + 1; j < wrap.size(); ++j) {
                if (bonded(*wrap[i], *wrap[j])) {
                    ADDONEEDGE(g.data(), i, j, m);
                }
            }
        }

        // Off-load to nauty!
        densenauty(g.data(), lab, ptn, orbits, &options, &stats, m, n,
                   cg.data());

        // Put atoms in nauty order
        transform_into(lab, lab + atoms.size(), order,
                       [&](int i) { return *wrap[i]; });

        // fill colour data
        std::transform(lab, lab + atoms.size(), cg.kinds().begin(),
                       [&](int i) { return wrap[i].toInt(); });

        // Reverse such that higher coordination near centre
        // could use { return atoms[atoms.size() - 1 - i]; }
        // std::reverse(order.begin(), order.end());

        // makeFirstOrigin(order);

        return cg;
    }
};
