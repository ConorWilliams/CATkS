#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#define MAXN 128

#include "MurmurHash3.h"
#include "nautinv.h"
#include "nauty.h"
#include "nlohmann/json.hpp"

#include "Catalog.hpp"
#include "utils.hpp"

static_assert(MAXM * 64 == MAXN, "Nauty is playing up!");

struct NautyGraph {
  private:
    std::array<graph, MAXN * MAXM> g{};
    std::array<int, MAXN> k;

  public:
    inline graph const *data() const { return g.data(); }
    inline graph *data() { return g.data(); }

    inline auto &kinds() { return k; }

    // Could be improved by storing the hash in the class and only computing
    // when g/k changes.
    std::array<std::uint64_t, 2> hash() const {
        std::array<std::uint64_t, 2> h;
        MurmurHash3_x86_128(this, sizeof(NautyGraph), 0, h.data());
        return h;
    }

    std::string to_string() const {
        std::string str = "rdf.";

        auto h = hash();

        str += std::to_string(h[0]) + std::to_string(h[1]);

        for (auto &&elem : g) {
            str += std::to_string(elem);
        }
        for (auto &&elem : k) {
            str += std::to_string(elem);
        }

        return str.substr(0, 255);
    }

    friend inline bool operator==(NautyGraph const &a, NautyGraph const &b) {
        return a.g == b.g && a.k == b.k;
    }

    friend void to_json(nlohmann::json &j, NautyGraph const &graph) {
        j = nlohmann::json{
            {"g", graph.g},
            {"k", graph.k},
        };
    }

    friend void from_json(nlohmann::json const &j, NautyGraph &graph) {
        j.at("g").get_to(graph.g);
        j.at("k").get_to(graph.k);
    }
};

// specialising for std::unordered
namespace std {
template <> struct hash<NautyGraph> {
    std::size_t operator()(NautyGraph const &x) const {
        auto h = x.hash();
        return h[0] ^ h[1];
    }
};
} // namespace std

// Find cental atom and move it to front, this is because the cental
// atom is best suited as the origin of the soon to be defined local
// coordinate system during findBasis() whic uses order[0] as origin.
template <typename Atom_t> void makeFirstOrigin(std::vector<Atom_t> &ordered) {

    Eigen::Vector3d sum = std::accumulate(
        ordered.begin(), ordered.end(), Eigen::Vector3d{0, 0, 0},
        [](auto s, Atom_t const &atom) -> Eigen::Vector3d {
            return s + atom.pos();
        });

    sum /= ordered.size();

    auto centre = std::min_element(ordered.begin(), ordered.end(),
                                   [&](Atom_t const &a, Atom_t const &b) {
                                       return (a.pos() - sum).squaredNorm() <
                                              (b.pos() - sum).squaredNorm();
                                   });

    std::iter_swap(ordered.begin(), centre);

    return;
}

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

    static constexpr int SHIFT = 1; // kind in {0,1} therfore need 1-bit
    static constexpr double GRANULARITY = 0.1; // approx DIST_TOL / sqrt(3)

    AtomWrap(Atom_t &atom) : atom{&atom}, sum{0} {}

    inline Atom_t *operator->() { return atom; }
    inline Atom_t const *operator->() const { return atom; }

    inline Atom_t &operator*() { return *atom; }
    inline Atom_t const &operator*() const { return *atom; }

    inline bool operator<(AtomWrap const &other) {
        if (atom->kind() == other->kind()) {
            return sum2Int() > other.sum2Int();
        } else {
            return atom->kind() > other->kind();
        }
    }

    inline bool operator==(AtomWrap const &other) {
        return atom->kind() == other->kind() && sum2Int() == other.sum2Int();
    }

    inline bool operator!=(AtomWrap const &other) { return !(*this == other); }

    inline int colour() {
        CHECK(sum2Int() < (INT_MAX >> SHIFT), "overflow");
        CHECK(atom->kind() < (1 << SHIFT), "kind too large");
        return (sum2Int() << SHIFT) ^ atom->kind();
    }
};

class NautyCanon2 {
  private:
    enum : bool { colour = false, plain = true };

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

        constexpr std::size_t MIN_NEIGH = 6; // >3 for triangulation
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
                    // Greater than as self != neighbour
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
