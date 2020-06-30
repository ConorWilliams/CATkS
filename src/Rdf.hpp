#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "MurmurHash3.h"
#include "nlohmann/json.hpp"

#include "utils.hpp"

inline constexpr double R_TOPO = 6; // angstrom
inline constexpr std::size_t R_BINS = 64;
inline constexpr std::size_t MAX_ATOMS = 80;

inline constexpr double R_BIN_WIDTH = 2 * R_TOPO / R_BINS;
inline constexpr double INV_R_BIN_WIDTH = 1 / R_BIN_WIDTH;

struct RDFGraph {
  private:
    std::array<uint8_t, R_BINS> rdf{};
    std::array<int, MAX_ATOMS> k{};

  public:
    inline auto &operator[](std::size_t i) { return rdf[i]; }
    inline auto &kinds() { return k; }

    // Could be improved by storing the hash in the class and only computing
    // when g/k changes.
    std::array<std::uint64_t, 2> hash() const {
        std::array<std::uint64_t, 2> h;
        MurmurHash3_x86_128(this, sizeof(RDFGraph), 0, h.data());
        return h;
    }

    friend inline bool operator==(RDFGraph const &a, RDFGraph const &b) {
        return a.rdf == b.rdf && a.k == b.k;
    }

    friend void to_json(nlohmann::json &j, RDFGraph const &graph) {
        j = nlohmann::json{
            {"rdf", graph.rdf},
            {"k", graph.k},
        };
    }

    friend void from_json(nlohmann::json const &j, RDFGraph &graph) {
        j.at("rdf").get_to(graph.rdf);
        j.at("k").get_to(graph.k);
    }
};

// specialising for std::unordered
namespace std {
template <> struct hash<RDFGraph> {
    std::size_t operator()(RDFGraph const &x) const {
        auto h = x.hash();
        return h[0] ^ h[1];
    }
};
} // namespace std

namespace detail {

bool are_close(double a, double b, double tol = 0.5 * R_BIN_WIDTH) {
    double dif = std::abs(a - b);
    double avg = 0.5 * (a + b);
    return dif / avg < tol ? true : false;
}

template <typename Atom_t> struct Wrap {
    // BOD
    Atom_t const &atom;
    std::vector<double> mem;

    Wrap(Atom_t const &atom) : atom{atom}, mem{} {}

    friend inline bool operator==(Wrap const &a, Wrap const &b) {
        check(a.mem.size() == b.mem.size(), "== comparing diff lengths");

        for (std::size_t i = 0; i < a.mem.size(); ++i) {
            std::size_t j = a.mem.size() - 1 - i;
            if (!are_close(a.mem[j], b.mem[j])) {
                return false;
            }
        }

        return true;
    }

    friend inline bool operator<(Wrap const &a, Wrap const &b) {

        check(a.mem.size() == b.mem.size(), "< comparing diff lengths");

        for (std::size_t i = 0; i < a.mem.size(); ++i) {
            std::size_t j = a.mem.size() - 1 - i;
            if (!are_close(a.mem[j], b.mem[j])) {
                return a.mem[j] < b.mem[j];
            }
        }

        // equal is not less!
        return false;
    }
};

template <typename It> It firstUnique(It begin, It end) {

    auto start = begin;
    auto probe = begin;

    while (start != end) {
        if (probe != end && *start == *probe) {
            ++probe;
        } else {
            if (std::next(start) == probe) {
                return start;
            } else {
                start = probe;
            }
        }
    }

    return end;
}

} // namespace detail

class RDFCanon {
  public:
    using Key_t = RDFGraph;

    // NOT thread safe.
    // Writes canonically ordered atoms to order and returns the canonically
    // ordered graph represented as a dense adjecency matrix.
    template <typename Atom_t>
    static inline RDFGraph canonicalize(std::vector<Atom_t> &atoms,
                                        std::vector<Atom_t> &order) {

        check(order.size() == 0, "order has atoms already");
        check(atoms.size() <= MAX_ATOMS, "too many atoms");

        RDFGraph canon{};

        std::vector<detail::Wrap<Atom_t>> wrap{atoms.begin(), atoms.end()};

        //
        for (auto &&a : wrap) {
            for (auto const &b : wrap) {
                if (&a != &b) {
                    double dist = (a.atom.pos() - b.atom.pos()).norm();
                    a.mem.push_back(dist);

                    check(dist < 2 * R_TOPO, "R_topo no aligned");
                    canon[dist * INV_R_BIN_WIDTH] += 1;
                }
            }
            std::sort(a.mem.begin(), a.mem.end());
        }

        for (auto it = wrap.begin(); it != wrap.end(); ++it) {

            std::sort(it, wrap.end());

            auto pivot = firstUnique(it, wrap.end());

            if (pivot != wrap.end()) {
                std::iter_swap(it, pivot);
            }

            for (auto rem = std::next(it); rem != wrap.end(); ++rem) {
                rem->mem.push_back((rem->atom.pos() - it->atom.pos()).norm());
            }
        }

        return {};
    }
};
