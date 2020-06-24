#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "DumpXYX.hpp"
#include "Nauty.hpp"
#include "utils.hpp"

struct AtomPlus {
    Eigen::Vector3d pos{0, 0, 0};
    long idx{};
};

// Build a list of atoms near atom at index centre in x.
template <typename T>
std::vector<AtomPlus> findNeighAtoms(Vector const &x, std::size_t centre,
                                     T const &f) {
    std::vector<AtomPlus> neigh;

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
Eigen::Matrix3d findBasis(std::vector<AtomPlus> const &list, T const &f) {

    Eigen::Vector3d origin = list[0].pos;
    Eigen::Matrix3d basis;

    // std::cout << "[\nAtom count: " << list.size() << std::endl;

    check(list.size() > 2, "Not enough atoms to define basis");

    basis.col(0) = f.minImage(list[1].pos - origin).normalized();

    std::cout << "e0 @ 0->" << 1 << std::endl;

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
        std::cerr << "All atoms colinear" << std::endl;
        std::terminate();
    }();

    for (std::size_t i = index_e1 + 1; i < list.size(); ++i) {
        Eigen::Vector3d e2 = f.minImage(list[i].pos - origin).normalized();

        double triple_prod = std::abs(basis.col(2).adjoint() * e2);

        // check for coplanarity with e0, e1
        if (triple_prod > 0.1) {
            basis.col(2) = e2;
            std::cout << "e2 @ 0->" << i << std::endl;
            // std::cout << "Triple_prod " << triple_prod << std::endl;
            break;
        }
    }

    // std::cout << "\nTransformer (non-orthoganlaised) \n" << basis <<
    // std::endl;

    return modifiedGramSchmidt(basis);
}

inline constexpr double BOND_DISTANCE = 3.0; // angstrom

template <typename T>
std::vector<AtomPlus> canonicalOrder(std::vector<AtomPlus> const &list,
                                     T const &f) {
    NautyGraph graph(list.size());

    for (std::size_t i = 0; i < list.size(); ++i) {
        for (std::size_t j = i + 1; j < list.size(); ++j) {
            double sqdist = f.minImage(list[i].pos - list[j].pos).squaredNorm();
            if (sqdist < BOND_DISTANCE * BOND_DISTANCE) {
                graph.addEdge(i, j);
            }
        }
    }

    int const *order = graph.getCanonical();

    std::vector<AtomPlus> ordered;

    for (std::size_t i = 0; i < list.size(); ++i) {
        ordered.push_back(list[order[i]]);
    }

    return ordered;
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

    std::vector<AtomPlus> near = findNeighAtoms(init, centre, f);

    // std::vector<int> col(init.size() / 3, 0);
    // for (std::size_t i = 0; i < near.size(); ++i) {
    //     col[near[i].idx] = 1;
    // }
    // output(init, col);

    //////// find canonical-ordering ///////////

    near = canonicalOrder(near, f);

    // for (std::size_t i = 0; i < near.size(); ++i) {
    //     col[near[i].idx] = i;
    // }
    //
    // output(init, col);

    ///////////// get basis vectors /////////////

    Eigen::Matrix3d transform = findBasis(near, f);

    std::vector<Eigen::Vector3d> reference;

    std::transform(near.begin(), near.end(), std::back_inserter(reference),
                   [&](AtomPlus const &atom) -> Eigen::Vector3d {
                       Eigen::Vector3d delta = {
                           end[3 * atom.idx + 0] - init[3 * atom.idx + 0],
                           end[3 * atom.idx + 1] - init[3 * atom.idx + 1],
                           end[3 * atom.idx + 2] - init[3 * atom.idx + 2],
                       };

                       return transform.transpose() * f.minImage(delta);
                   });

    // std::cout << "First atom delta: " << reference[0][0] << ' '
    //           << reference[0][1] << ' ' << reference[0][2] << "\n]\n";

    return {centre, std::move(reference)};
}

template <typename T>
Vector reconstruct(Vector const &init, std::size_t centre,
                   std::vector<Eigen::Vector3d> const &ref, T const &f) {
    Vector end = init;

    std::vector<AtomPlus> near = findNeighAtoms(init, centre, f);

    check(near.size() == ref.size(), "wrong num atoms in reconstruction");

    near = canonicalOrder(near, f);

    // std::vector<int> col(init.size() / 3, 0);
    // for (std::size_t i = 0; i < near.size(); ++i) {
    //     check(near[i].idx < (int)col.size(), "bug in ordering alg again");
    //     col[near[i].idx] = i + 1;
    // }
    //
    // output(init, col);

    Eigen::Matrix3d transform = findBasis(near, f);

    for (std::size_t i = 0; i < near.size(); ++i) {
        Eigen::Vector3d delta = transform * ref[i];

        end[3 * near[i].idx + 0] += delta[0];
        end[3 * near[i].idx + 1] += delta[1];
        end[3 * near[i].idx + 2] += delta[2];
    }

    // output(end, col);

    return end;
}

template <typename T>
std::vector<Eigen::Vector3d> classifyTopo(Vector const &x, std::size_t idx,
                                          T const &f) {

    std::vector<AtomPlus> near = findNeighAtoms(x, idx, f);

    std::vector<int> col(x.size() / 3, 0);
    // for (std::size_t i = 0; i < near.size(); ++i) {
    //     col[near[i].idx] = 1;
    // }
    // output(init, col);

    //////// find canonical-ordering ///////////

    near = canonicalOrder(near, f);

    for (std::size_t i = 0; i < near.size(); ++i) {
        col[near[i].idx] = i;
    }
    col[near[0].idx] = 99;
    output(x, col);

    ///////////// get basis vectors /////////////

    Eigen::Matrix3d transform = findBasis(near, f);

    std::vector<Eigen::Vector3d> reference;

    Eigen::Vector3d origin = near[0].pos;

    std::transform(near.begin(), near.end(), std::back_inserter(reference),
                   [&](AtomPlus const &atom) -> Eigen::Vector3d {
                       Eigen::Vector3d delta = atom.pos - origin;

                       return transform.transpose() * f.minImage(delta);
                   });

    return reference;
}
