#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "Canon.hpp"

#include "nautinv.h"

// pre H was working at 2.55, 2.7 //
static constexpr double F_F_BOND = 2.67; // 2.47 -- 2.86 angstrom
static constexpr double H_H_BOND = 3.00; // 0.25 0.5 0.75 vacancy neigh
static constexpr double F_H_BOND = 3.10; //

static const Eigen::Matrix2d DISTS{
    {F_F_BOND, F_H_BOND},
    {F_H_BOND, H_H_BOND},
};

template <typename Atom_t>
inline static bool bonded(Atom_t const &a, Atom_t const &b) {

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
      private:
        inline int sum2Int() const {
            // Convert sum into an int storing kind in lower order bits
            CHECK(sum < GRANULARITY * INT_MAX, "overflow gonna getcha");

            return static_cast<int>(sum / GRANULARITY);
        };

      public:
        Atom_t *atom;
        double sum;

        static constexpr int SHIFT = 1;
        static constexpr double GRANULARITY = 0.1;

        AtomWrap(Atom_t &atom) : atom{&atom}, sum{0} {}

        inline Atom_t *operator->() { return atom; }
        inline Atom_t const *operator->() const { return atom; }

        inline Atom_t &operator*() { return *atom; }
        inline Atom_t const &operator*() const { return *atom; }

        inline bool operator<(AtomWrap const &other) {
            if (atom->kind() == other->kind()) {
                return sum2Int() > other.sum2Int();
            } else {
                return atom->kind() < other->kind();
            }
        }

        inline bool operator==(AtomWrap const &other) {
            return atom->kind() == other->kind() &&
                   sum2Int() == other.sum2Int();
        }

        inline bool operator!=(AtomWrap const &other) {
            return !(*this == other);
        }

        inline int colour() {
            CHECK(sum2Int() < (INT_MAX >> SHIFT), "overflow");
            CHECK(atom->kind() < (1 << SHIFT), "kind too large");
            return (sum2Int() << SHIFT) ^ atom->kind();
        }
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
        static DEFAULTOPTIONS_DIGRAPH(options);

        options.getcanon = true;

        NautyGraph g{};
        NautyGraph cg{};

        std::size_t n = atoms.size();
        std::size_t m = SETWORDSNEEDED(n);

        CHECK(n <= MAXN && m <= MAXM, "too many atoms " << n << ' ' << m);
        nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

        //// using coloured graph mode
        options.defaultptn = colour;

        std::vector<AtomWrap<Atom_t>> wrap{atoms.begin(), atoms.end()};

        constexpr std::size_t MIN_NEIGH = 4;
        constexpr /**/ std::size_t BINS = 24;
        constexpr /*     */ double RMAX = 2 * 6; // diametre 2*RCUT

        for (std::size_t i = 0; i < n; ++i) {
            std::array<std::size_t, BINS> rdf{};

            for (std::size_t j = 0; j < n; ++j) {
                ++rdf[(wrap[i]->pos() - wrap[j]->pos()).norm() * 2];
            }

            const double bond = (RMAX / BINS) * [&]() -> std::size_t {
                std::size_t count = 0;
                for (std::size_t i = 0; i < rdf.size(); ++i) {
                    if (count > MIN_NEIGH) {
                        return i;
                    }
                    count += rdf[i];
                }
                return -1;
            }();

            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) {
                    double dist = (wrap[i]->pos() - wrap[j]->pos()).norm();
                    if (dist < bond) {
                        wrap[i].sum += dist;
                        ADDONEARC(g.data(), i, j, m);
                    }
                }
            }
        }

        std::iota(lab, lab + n, 0); // TODO : only needs to be done once

        std::sort(lab, lab + n,
                  [&](int a, int b) { return wrap[a] < wrap[b]; });

        for (std::size_t i = 0; i < n; ++i) {
            if (i + 1 == n || wrap[lab[i + 1]] != wrap[lab[i]]) {
                ptn[i] = 0;
            } else {
                ptn[i] = 1;
            }
        }
        ////

        // Off-load to nauty!
        densenauty(g.data(), lab, ptn, orbits, &options, &stats, m, n,
                   cg.data());

        for (std::size_t i = 0; i < n; ++i) {
            order.push_back(*wrap[lab[n - 1 - i]]); // reverses order
            cg.kinds()[i] = wrap[lab[i]].colour();
        }

        makeFirstOrigin(order);

        return cg;
    }
};
