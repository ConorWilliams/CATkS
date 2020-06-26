#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <iostream>

#define MAXN 128

#include "MurmurHash3.h"
#include "nauty.h"
#include "nlohmann/json.hpp"
#include "utils.hpp"

static_assert(MAXM == 2, "Nauty is playing up!");

struct Graph {
  private:
    std::array<graph, MAXN * MAXM> g{};

  public:
    inline graph const *data() const { return g.data(); }
    inline graph *data() { return g.data(); }

    std::array<std::uint64_t, 2> hash() const {

        std::array<std::uint64_t, 2> h;
        MurmurHash3_x86_128(g.data(), sizeof(graph) * MAXN * MAXM, 0, h.data());

        return h;
    }

    std::string to_string() const {
        std::string str = "rdf.";

        auto h = hash();

        str += std::to_string(h[0]) + std::to_string(h[1]);

        for (auto &&elem : g) {
            str += std::to_string(elem);
        }

        return str.substr(0, 255);
    }

    friend inline bool operator==(Graph a, Graph b) { return a.g == b.g; }

    friend void to_json(nlohmann::json &j, Graph const &graph) {
        j = nlohmann::json{graph.g};
    }

    friend void from_json(nlohmann::json const &j, Graph &graph) {
        j.at(0).get_to(graph.g);
    }
};

// specialising for std::unordered
namespace std {
template <> struct hash<Graph> {
    std::size_t operator()(Graph const &x) const {
        auto h = x.hash();
        return h[0] ^ h[1];
    }
};
} // namespace std

// NOT thread safe
// Writes canonically ordered atoms to order and returns the canonically ordered
// graph represented as a dense adjecency matrix
template <typename Atom_t>
inline Graph canonicalize(std::vector<Atom_t> const &atoms,
                          std::vector<Atom_t> &order) {

    check(order.size() == 0, "order has atoms already");

    static constexpr double BOND_DISTANCE = 2.52; // angstrom 2.66

    Graph g{};
    Graph cg{};

    static int lab[MAXN];
    static int ptn[MAXN];
    static int orbits[MAXN];

    static DEFAULTOPTIONS_GRAPH(options);
    static statsblk stats;

    options.getcanon = true;
    options.defaultptn = true; // change for coloured graphs

    std::size_t n = atoms.size();
    std::size_t m = SETWORDSNEEDED(n);

    check(n <= MAXN && m <= MAXM, "too many atoms in list" << n << ' ' << m);

    nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {

            double sqdist = (atoms[i].pos() - atoms[j].pos()).squaredNorm();

            if (sqdist < BOND_DISTANCE * BOND_DISTANCE) {
                ADDONEEDGE(g.data(), i, j, m);
            }
        }
    }

    densenauty(g.data(), lab, ptn, orbits, &options, &stats, m, n, cg.data());

    transform_into(lab, lab + atoms.size(), order,
                   [&](int i) { return atoms[i]; });

    return cg;
}
