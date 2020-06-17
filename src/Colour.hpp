#pragma once

#include <array>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include "Eigen/Eigenvalues"

#include "MurmurHash3.h"
#include "utils.hpp"

using sint_t = uint8_t;
inline constexpr sint_t RADIAL_BINS = 32; // power of 2 for fast division
inline constexpr double BIN_WIDTH = 1. / RADIAL_BINS;

inline constexpr double NEG_TOL = -0.01;

#include "DumpXYX.hpp"

int FRAME = 0;
static const std::string head{"/home/cdt1902/dis/CATkS/plt/dump/all_"};
static const std::string tail{".xyz"};

template <typename T = std::vector<int>>
void output(Vector const &x, T const &kinds) {
    dumpXYX(head + std::to_string(FRAME++) + tail, x, kinds);
}

void output(Vector const &x) { output(x, std::vector<int>(x.size() / 3, 0)); }

class Rdf {
  private:
    std::array<sint_t, RADIAL_BINS> rdf{};

  public:
    Rdf() = default;

    inline void add(double r) {
        check(r >= 0 && r < 1, "not on unit interval");

        std::size_t bin = r * RADIAL_BINS;
        ++rdf[bin];

        // // bin fuzzing
        // if ((r + BIN_WIDTH / 4) * RADIAL_BINS != bin) {
        //     ++rdf[bin + 1];
        // } else if ((r - BIN_WIDTH / 4) * RADIAL_BINS != bin) {
        //     ++rdf[bin - 1];
        // }
    }

    inline sint_t const *data() const { return rdf.data(); }

    friend inline bool operator==(Rdf a, Rdf b) { return a.rdf == b.rdf; }
};

// specialising for std::unordered set
namespace std {
template <> struct hash<Rdf> {
    std::size_t operator()(Rdf const &x) const {
        static_assert(sizeof(std::size_t) * 8 == 64, "need 64bit std::size_t");

        std::size_t hash_out[2];
        MurmurHash3_x86_128(x.data(), sizeof(x), 0, hash_out);
        return hash_out[0] ^ hash_out[1];
    }
};
} // namespace std

struct Topo {
    Eigen::Vector3d pos;
    int idx;
    std::vector<double> mem = {};

    friend inline bool operator==(Topo const &a, Topo const &b) {
        if (a.mem.size() != b.mem.size()) {
            return false;
        }

        for (std::size_t i = 0; i < a.mem.size(); ++i) {
            std::size_t j = a.mem.size() - 1 - i;
            if (std::abs(a.mem[j] - b.mem[j]) > 0.1) {
                return false;
            }
        }

        return true;
    }

    friend inline bool operator<(Topo const &a, Topo const &b) {
        if (a.mem.size() < b.mem.size()) {
            return true;
        } else if (a.mem.size() > b.mem.size()) {
            return false;
        }

        for (std::size_t i = 0; i < a.mem.size(); ++i) {
            std::size_t j = a.mem.size() - 1 - i;
            if (std::abs(a.mem[j] - b.mem[j]) > 0.1) {
                return a.mem[j] < b.mem[j];
            }
        }

        return true;
    }
};

bool near(double x, double y) { return std::abs(x - y) < 0.01; }

// Build a list of atoms near atom at index centre in x.
template <typename T>
std::vector<Topo> findNeighAtoms(Vector const &x, std::size_t centre,
                                 T const &f) {
    std::vector<Topo> neigh;

    double cx = x[3 * centre + 0];
    double cy = x[3 * centre + 1];
    double cz = x[3 * centre + 2];

    for (int i = 0; i < x.size(); i = i + 3) {
        double dist_sq =
            f.periodicNormSq(cx, cy, cz, x[i + 0], x[i + 1], x[i + 2]);

        if (dist_sq < f.rcut() * f.rcut()) {
            neigh.push_back({{x[i + 0], x[i + 1], x[i + 2]}, i / 3});
        }
    }

    return neigh;
}

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

// sorts a list of Topos into a canonicle ordering in O(n^3 ln n)
template <typename T> void canonicalOrder(std::vector<Topo> &list, T const &f) {
    // Compute sums for each atom
    for (auto &&atom : list) {
        atom.mem.push_back(0);
        for (auto &&other : list) {
            atom.mem.back() += f.minImage(atom.pos - other.pos).squaredNorm();
        }
    }

    // std::cout << std::fixed;
    // std::cout << std::setprecision(0);
    //
    // for (auto &&a : list) {
    //     std::cout << a.idx << ':' << a.mem[0] << ' ';
    // }
    // std::cout << std::endl;

    // O(n)
    for (auto it = list.begin(); it != list.end(); ++it) {

        // O(n log n) comparisons at O(n) per comparison = O(n^2 ln n)
        std::sort(it, list.end());

        if (auto pivot = firstUnique(it, list.end()); pivot != list.end()) {
            std::iter_swap(it, pivot);
        } else {
            std::cout << "No unique pivot" << std::endl;
        }

        // std::cout << "Pivot was: " << it->idx << std::endl;

        for (auto rem = std::next(it); rem != list.end(); ++rem) {
            rem->mem.push_back(f.minImage(rem->pos - it->pos).squaredNorm());
        }
    }
}

// sorts a list of Topos into a canonicle ordering in O(n^2 ln n)
template <typename T>
void canonicalOrder2(std::vector<Topo> &list, T const &f) {
    // Compute sums for each atom
    for (auto &&atom : list) {
        for (auto &&other : list) {
            atom.mem.push_back(f.minImage(atom.pos - other.pos).squaredNorm());
        }
        std::sort(atom.mem.begin(), atom.mem.end());
    }

    for (auto it = list.begin(); it != list.end(); ++it) {

        auto min = std::min_element(it, list.end());

        std::iter_swap(it, min);

        for (auto rem = std::next(it); rem != list.end(); ++rem) {
            rem->mem.push_back(f.minImage(rem->pos - it->pos).squaredNorm());
        }
    }
}

Eigen::Matrix3d modifiedGramSchmidt(Eigen::Matrix3d const &in) {
    Eigen::Matrix3d out;

    out.col(0) = in.col(0);
    check(out.col(0).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(0).normalize();

    out.col(1) = in.col(1) - (in.col(1).adjoint() * out.col(0)) * out.col(0);
    check(out.col(1).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(1).normalize();

    out.col(2) = in.col(2) - (in.col(2).adjoint() * out.col(0)) * out.col(0);
    out.col(2) -= (out.col(2).adjoint() * out.col(1)) * out.col(1);
    check(out.col(2).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(2).normalize();

    using std::abs;

    check(abs(out.col(0).transpose() * out.col(1)) < 0.01, "gram schmidt fail");
    check(abs(out.col(1).transpose() * out.col(2)) < 0.01, "gram schmidt fail");
    check(abs(out.col(2).transpose() * out.col(0)) < 0.01, "gram schmidt fail");

    return out;
}

template <typename T>
Eigen::Matrix3d findBasis(std::vector<Topo> const &list, T const &f) {

    Eigen::Vector3d origin = list[0].pos;
    Eigen::Matrix3d basis;

    std::cout << "Atom count: " << list.size() << std::endl;
    check(list.size() > 2, "not enough atoms to define basis");

    basis.col(0) = f.minImage(list[1].pos - origin).normalized();

    std::cout << "e1 @ 0->" << 1 << std::endl;

    std::size_t index_e1 = [&]() {
        for (std::size_t i = 2; i < list.size(); ++i) {
            Eigen::Vector3d e1 = f.minImage(list[i].pos - origin).normalized();

            Eigen::Vector3d cross = basis.col(0).cross(e1);

            // check for colinearity
            if (cross.squaredNorm() > 0.1) {
                basis.col(1) = e1;
                basis.col(2) = cross;
                std::cout << "e1 @ 0->" << i << std::endl;
                return i;
            }
        }
        std::cerr << "all atoms colinear" << std::endl;
        std::terminate();
    }();

    for (std::size_t i = index_e1 + 1; i < list.size(); ++i) {
        Eigen::Vector3d e2 = f.minImage(list[i].pos - origin).normalized();

        double triple_prod = std::abs(basis.col(2).adjoint() * e2);

        // check for coplanarity with e0, e1
        if (triple_prod > 0.1) {
            basis.col(2) = e2;
            std::cout << "e2 @ 0->" << i << std::endl;
            std::cout << "triple_prod " << triple_prod << std::endl;
            break;
        }
    }

    std::cout << "\nTransformer (non-orthoganlaised) \n" << basis << std::endl;

    return modifiedGramSchmidt(basis);
}

// Returns the index of the central atom for the mechanisim as well as the
// reference data to reconstruct mechanisim.
template <typename T>
std::tuple<std::size_t, std::vector<Eigen::Vector3d>>
classifyMech(Vector const &init, Vector const &end, T const &f) {
    std::size_t centre = 0;

    { // find furthest moved
        double dr_sq_max = 0;

        for (int i = 0; i < end.size(); i += 3) {
            double dr_sq =
                f.periodicNormSq(end[i + 0], end[i + 1], end[i + 2],
                                 init[i + 0], init[i + 1], init[i + 2]);

            if (dr_sq > dr_sq_max) {
                centre = i / 3;
                dr_sq_max = dr_sq;
            }
        }
    }

    std::vector<Topo> near = findNeighAtoms(init, centre, f);

    std::vector<int> col(init.size() / 3, 0);
    for (std::size_t i = 0; i < near.size(); ++i) {
        col[near[i].idx] = 1;
    }
    output(init, col);

    //////// find canonical-ordering ///////////

    canonicalOrder2(near, f);

    for (std::size_t i = 0; i < near.size(); ++i) {
        col[near[i].idx] = i + 1;
    }
    output(init, col);

    ///////////// get basis vectors /////////////

    Eigen::Matrix3d transform = findBasis(near, f);

    std::vector<Eigen::Vector3d> reference;

    ////// store normalised /////
    // std::cout << std::endl << "memory" << std::endl;
    for (auto &&atom : near) {
        Eigen::Vector3d delta;

        delta[0] = end[3 * atom.idx + 0] - init[3 * atom.idx + 0];
        delta[1] = end[3 * atom.idx + 1] - init[3 * atom.idx + 1];
        delta[2] = end[3 * atom.idx + 2] - init[3 * atom.idx + 2];

        delta = transform.transpose() * f.minImage(delta);

        // std::cout << delta[0] << ' ' << delta[1] << ' ' << delta[2]
        //           << std::endl;

        reference.push_back(delta);
    }

    return {centre, std::move(reference)};
}

template <typename T>
Vector reconstruct(Vector const &init, std::size_t centre,
                   std::vector<Eigen::Vector3d> const &ref, T const &f) {
    Vector end = init;

    std::vector<Topo> near = findNeighAtoms(init, centre, f);

    check(near.size() == ref.size(), "wrong num atoms in reconstruction");

    canonicalOrder2(near, f);

    std::vector<int> col(init.size() / 3, 0);
    for (std::size_t i = 0; i < near.size(); ++i) {
        col[near[i].idx] = i + 1;
    }

    output(init, col);

    Eigen::Matrix3d transform = findBasis(near, f);

    for (std::size_t i = 0; i < near.size(); ++i) {
        Eigen::Vector3d delta = transform * ref[i];

        end[3 * near[i].idx + 0] += delta[0];
        end[3 * near[i].idx + 1] += delta[1];
        end[3 * near[i].idx + 2] += delta[2];
    }

    output(end, col);

    return end;
}
