#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#define MAXN 128

#include "MurmurHash3.h"
#include "nauty.h"
#include "nlohmann/json.hpp"

#include "utils.hpp"

static_assert(MAXM == 2, "Nauty is playing up!");

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

class NautyCanon {
  private:
    enum : bool { colour = false, plain = true };

    // pre H was working at 2.55 // 2.47 -- 2.86 angstrom (just > first neigh)
    static constexpr double F_F_BOND = 2.55;
    static constexpr double H_H_BOND = 2.00; // 0.5^2+0.5^2 vacancy neigh
    static constexpr double F_H_BOND = 2.65; // 0.75^2 + 0.5^2

  public:
    template <typename Atom_t>
    inline static bool bonded(Atom_t const &a, Atom_t const &b) {

        static const Eigen::Matrix2d DISTS{
            {F_F_BOND, F_H_BOND},
            {F_H_BOND, H_H_BOND},
        };

        CHECK(a.kind() < 2, "atom type not valid " << a.kind());
        CHECK(b.kind() < 2, "atom type not valid " << b.kind());

        double sqdist = (a.pos() - b.pos()).squaredNorm();

        return sqdist < DISTS(a.kind(), b.kind()) * DISTS(a.kind(), b.kind());
    }

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

        CHECK(n <= MAXN && m <= MAXM, "too many atoms" << n << ' ' << m);

        nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

        //// order by colour
        options.defaultptn = colour;

        std::sort(atoms.begin(), atoms.end(),
                  [](Atom_t const &a, Atom_t const &b) {
                      return a.kind() < b.kind();
                  });

        for (std::size_t i = 0; i < n; ++i) {
            lab[i] = i;
            if (i + 1 == n || atoms[i + 1].kind() != atoms[i].kind()) {
                ptn[i] = 0;
            } else {
                ptn[i] = 1;
            }
        }
        ////

        // Fill in adjecency matrix
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            for (std::size_t j = i + 1; j < atoms.size(); ++j) {
                if (bonded(atoms[i], atoms[j])) {
                    ADDONEEDGE(g.data(), i, j, m);
                }
            }
        }

        // Off-load to nauty!
        densenauty(g.data(), lab, ptn, orbits, &options, &stats, m, n,
                   cg.data());

        // Put atoms in nauty order
        transform_into(lab, lab + atoms.size(), order,
                       [&](int i) { return atoms[i]; });

        // fill colour data
        std::transform(lab, lab + atoms.size(), cg.kinds().begin(),
                       [&](int i) { return atoms[i].kind(); });

        // Reverse such that higher coordination near centre
        // could use { return atoms[atoms.size() - 1 - i]; }
        std::reverse(order.begin(), order.end());

        makeFirstOrigin(order);

        return cg;
    }
};
