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

#include "utils.hpp"

#include "DumpXYX.hpp"

bool are_close(double a, double b, double tol = 0.04) {
    double dif = std::abs(a - b);
    double avg = 0.5 * (a + b);
    return dif / avg < tol ? true : false;
}

struct AtomPlus {
    Eigen::Vector3d pos{0, 0, 0};
    long idx{};
    std::vector<double> mem{};

    friend inline bool operator==(AtomPlus const &a, AtomPlus const &b) {
        if (a.mem.size() != b.mem.size()) {
            return false;
        }

        for (std::size_t i = 0; i < a.mem.size(); ++i) {
            std::size_t j = a.mem.size() - 1 - i;
            if (!are_close(a.mem[j], b.mem[j])) {
                return false;
            }
        }

        return true;
    }

    friend inline bool operator<(AtomPlus const &a, AtomPlus const &b) {

        check(a.mem.size() == b.mem.size(), "comparing diff lengths");

        if (a.mem.size() < b.mem.size()) {
            return true;
        } else if (a.mem.size() > b.mem.size()) {
            return false;
        }

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

        if (dist_sq < 3 * 3) {
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

// sorts a list of AtomPluss into a canonicle ordering in O(n^3 ln n)
template <typename T>
void canonicalOrder(std::vector<AtomPlus> &list, T const &f) {
    //
    for (auto &&atom : list) {
        for (auto const &other : list) {
            if (&atom != &other) {
                atom.mem.push_back(f.minImage(atom.pos - other.pos).norm());
            }
        }
        std::sort(atom.mem.begin(), atom.mem.end());
    }

    // O(n)
    for (auto it = list.begin(); it != list.end(); ++it) {
        // O(n log n) comparisons at O(n) per comparison = O(n^2 ln n)
        std::sort(it, list.end());

        std::cout << "\nSTART" << std::endl;

        for (auto &&atom : list) {
            for (auto &&r : atom.mem) {
                std::cout << r << ' ';
            }
            std::cout << std::endl;
        }

        if (auto pivot = firstUnique(it, list.end()); pivot != list.end()) {
            std::cout << "Pivot: " << pivot - list.begin() << std::endl;
            std::iter_swap(it, pivot);
        }

        for (auto rem = std::next(it); rem != list.end(); ++rem) {
            rem->mem.push_back(f.minImage(rem->pos - it->pos).norm());
        }
    }
}

// sorts a list of AtomPluss into a canonicle ordering in O(n^2 ln n)
template <typename T>
void canonicalOrder2(std::vector<AtomPlus> &list, T const &f) {
    // Compute sums for each atom
    for (auto &&atom : list) {
        for (auto &&other : list) {
            atom.mem.push_back(f.minImage(atom.pos - other.pos).squaredNorm());
        }
        std::sort(atom.mem.begin(), atom.mem.end());
    }

    for (auto it = list.begin(); it != list.end(); ++it) {

        std::iter_swap(it, std::min_element(it, list.end()));

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
Eigen::Matrix3d findBasis(std::vector<AtomPlus> const &list, T const &f) {

    Eigen::Vector3d origin = list[0].pos;
    Eigen::Matrix3d basis;

    // std::cout << "[\nAtom count: " << list.size() << std::endl;

    check(list.size() > 2, "Not enough atoms to define basis");

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

    canonicalOrder(near, f);

    // for (std::size_t i = 0; i < near.size(); ++i) {
    //     col[near[i].idx] = i + 1;
    // }
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

    canonicalOrder(near, f);

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

// Returns the index of the central atom for the mechanisim as well as the
// reference data to reconstruct mechanisim.
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

    canonicalOrder(near, f);

    for (std::size_t i = 0; i < near.size(); ++i) {
        col[near[i].idx] = i + 1;
    }
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
