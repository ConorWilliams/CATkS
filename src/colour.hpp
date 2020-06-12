#pragma once

#include <array>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>

#include "Eigen/Eigenvalues"

#include "MurmurHash3.h"
#include "utils.hpp"

using sint_t = uint8_t;
inline constexpr sint_t RADIAL_BINS = 32; // power of 2 for fast division
inline constexpr double BIN_WIDTH = 1. / RADIAL_BINS;

inline constexpr double NEG_TOL = -0.01;

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
};

bool near(double x, double y) { return std::abs(x - y) < 0.01; }

template <typename T>
Eigen::Vector3d findCentre(std::vector<Topo> const &list, T const &f) {
    Eigen::Vector3d centre;
    double min = std::numeric_limits<double>::max();

    for (auto &&atom : list) {
        double sum = 0;
        for (auto &&other : list) {
            sum += f.minImage(atom.pos - other.pos).squaredNorm();
        }
        if (sum < min) {
            centre = atom.pos;
            min = sum;
        }
    }

    check(min != std::numeric_limits<double>::max(),
          "failed to find a centre?");

    return centre;
}

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

// Maps positions of atoms in list to relative positions in inerta basis.
// Sorts list into canonical order.
// Returns orthoganal transform matrix T sush that:
// T * x[inerta basis] = x[standard basis]
template <typename T>
Eigen::Matrix3d toInertaBasis(std::vector<Topo> &list, T const &f) {
    //
    std::cout << std::endl
              << "There are " << list.size() << " atoms" << std::endl;

    check(list.size() > 1, "not enough atoms");

    // find center atom
    Eigen::Vector3d centre = findCentre(list, f);

    // make inerta tensor (rel to centre) and map : pos -> rel_pos
    Eigen::Matrix3d I = Eigen::Matrix3d::Zero();

    for (auto &&atom : list) {
        atom.pos = f.minImage(atom.pos - centre);

        I.noalias() += atom.pos.squaredNorm() * Eigen::Matrix3d::Identity() -
                       atom.pos * atom.pos.transpose();
    }

    std::cout << std::endl << "Inerta tensor:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << I(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    // find eigen vectors and values
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(I);

    Eigen::Matrix3d eVecs = solver.eigenvectors();
    Eigen::Vector3d eVals = solver.eigenvalues();

    std::cout << std::endl << "Eigen values:" << std::endl;
    std::cout << eVals[0] << ' ' << eVals[1] << ' ' << eVals[2] << std::endl;

    std::cout << std::endl << "Eigen vectors:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << eVecs(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    // if symetric top move unique to front for better sorting
    if (near(eVals[0], eVals[1])) {
        eVecs.col(0).swap(eVecs.col(2));
        std::swap(eVals[0], eVals[2]);
        std::cout << "top: swap 0,2" << std::endl;
    }

    // calculate projection sums
    Eigen::Vector3d sums = Eigen::Vector3d::Zero();

    for (auto &&atom : list) {
        sums.noalias() += eVecs.transpose() * atom.pos;
    }

    std::cout << std::endl << "Projection sums:" << std::endl;
    std::cout << sums[0] << ' ' << sums[1] << ' ' << sums[2] << std::endl;

    // flip axis as appropriate
    if (sums[0] < NEG_TOL) {
        eVecs.col(0) *= -1;
        std::cout << "flip 0" << std::endl;
    }
    if (sums[1] < NEG_TOL) {
        eVecs.col(1) *= -1;
        std::cout << "flip 1" << std::endl;
    }
    if (sums[2] < NEG_TOL) {
        eVecs.col(2) *= -1;
        std::cout << "flip 2" << std::endl;
    }

    // correct for symetric tops by fixing handedness of basis
    std::cout << std::endl << "handedness" << std::endl;
    double hand = eVecs.col(0).dot(eVecs.col(1).cross(eVecs.col(2)));

    std::cout << hand << std::endl;

    if (near(eVals[1], eVals[2]) && hand < 0) {
        eVecs.col(1).swap(eVecs.col(2));
        std::swap(eVals[1], eVals[2]);
        std::cout << "Top: handedness corrected" << std::endl;
    }

    /// map : rel_pos -> [corrected]_eigen_basis rel
    for (auto &&atom : list) {
        atom.pos = eVecs.transpose() * atom.pos;
    }

    // lex sort
    std::sort(list.begin(), list.end(), [&](Topo a, Topo b) {
        if (near(a.pos[0], b.pos[0])) {
            if (near(a.pos[1], b.pos[1])) {
                if (near(a.pos[2], b.pos[2])) {
                    return false;
                } else {
                    return a.pos[2] < b.pos[2];
                }
            } else {
                return a.pos[1] < b.pos[1];
            }
        } else {
            return a.pos[0] < b.pos[0];
        }
    });

    return eVecs;
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

    // std::vector<int> col(init.size() / 3, 0);
    // col[centre] = 1;
    // output(end, col);

    std::vector<Topo> near_atoms = findNeighAtoms(init, centre, f);

    // for (std::size_t i = 0; i < near_atoms.size(); ++i) {
    //     col[near_atoms[i].idx] = 1;
    // }
    //
    // output(init, col);

    //////// find canonical-ordering ///////////

    Eigen::Matrix3d transform = toInertaBasis(near_atoms, f);

    // for (std::size_t i = 0; i < near_atoms.size(); ++i) {
    //     col[near_atoms[i].idx] = i + 1;
    // }
    //
    // output(init, col);

    std::vector<Eigen::Vector3d> reference;

    ////// store normalised /////
    for (auto &&atom : near_atoms) {
        atom.pos[0] = end[3 * atom.idx + 0] - init[3 * atom.idx + 0];
        atom.pos[1] = end[3 * atom.idx + 1] - init[3 * atom.idx + 1];
        atom.pos[2] = end[3 * atom.idx + 2] - init[3 * atom.idx + 2];

        atom.pos = transform.transpose() * f.minImage(atom.pos);

        reference.push_back(atom.pos);
    }

    std::cout << std::endl << "memory" << std::endl;
    for (auto &&atom : near_atoms) {
        std::cout << atom.pos[0] << ' ' << atom.pos[1] << ' ' << atom.pos[2]
                  << std::endl;
    }

    return {centre, std::move(reference)};
}

template <typename T>
Vector reconstruct(Vector const &init, std::size_t centre,
                   std::vector<Eigen::Vector3d> const &ref, T const &f) {
    Vector end = init;

    std::vector<Topo> near_atoms = findNeighAtoms(init, centre, f);

    check(near_atoms.size() == ref.size(), "wrong num atoms in reconstruction");

    Eigen::Matrix3d transform = toInertaBasis(near_atoms, f);

    for (std::size_t i = 0; i < near_atoms.size(); ++i) {
        Eigen::Vector3d delta = transform * ref[i];

        end[3 * near_atoms[i].idx + 0] += delta[0];
        end[3 * near_atoms[i].idx + 1] += delta[1];
        end[3 * near_atoms[i].idx + 2] += delta[2];
    }

    return end;
}
