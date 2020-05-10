#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "EAM.hpp"
#include "utils.hpp"

// Helper class holds atom in LIST array of LCL with index of neighbours
template <typename T> class Atom {
  private:
    T k;
    std::array<double, 3> r;
    std::size_t n;

  public:
    Atom(T k, double x, double y, double z) : k{k}, r{x, y, z} {}

    inline T const &kind() const { return k; }
    inline std::size_t &next() { return n; }
    inline std::size_t next() const { return n; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

// Contains info about simulation box and performs Lambda mapping
class Box {
  private:
    struct Lim {
        double min, max, len, inv;
        Lim(double min, double max)
            : min{min}, max{max}, len{max - min}, inv{1.0 / len} {}
    };

    double m_rcut;

    std::array<Lim, 3> m_limits;

    std::size_t mx, my, mz; // number of cells

    Eigen::Array<std::size_t, 26, Eigen::Dynamic> adjOff; // 3^3 - 1

    inline std::size_t map3To1(std::size_t i, std::size_t j,
                               std::size_t k) const {
        return i + j * mx + k * mx * my;
    }

  public:
    Box(double rcut, double xmin, double xmax, double ymin, double ymax,
        double zmin, double zmax)
        : m_rcut{rcut}, m_limits{{{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}} {
        // sanity
        check(xmax > xmin, "cannot have x min <= max");
        check(ymax > ymin, "cannot have y min <= max");
        check(zmax > zmin, "cannot have z min <= max");

        check(rcut > 0, "rcut is negative");

        check(xmax - xmin >= 3 * rcut, "box too small");
        check(ymax - ymin >= 3 * rcut, "box too small");
        check(zmax - zmin >= 3 * rcut, "box too small");

        // round down (toward zero) truncation, therefore boxes always biger
        // than rcut
        mx = static_cast<std::size_t>((xmax - xmin) / rcut);
        my = static_cast<std::size_t>((ymax - ymin) / rcut);
        mz = static_cast<std::size_t>((zmax - zmin) / rcut);

        adjOff.resize(26, mx * my * mz);

        std::cout << "shape " << mx << ' ' << my << ' ' << mz << std::endl;

        for (std::size_t K = 0; K < mz; ++K) {
            for (std::size_t J = 0; J < my; ++J) {
                for (std::size_t I = 0; I < mx; ++I) {
                    // compute neighbour offsets
                    long idx = 0;
                    for (auto k : {-1, 0, 1}) {
                        for (auto j : {-1, 0, 1}) {
                            for (auto i : {-1, 0, 1}) {
                                if (i != 0 || j != 0 || k != 0) {
                                    adjOff(idx++, map3To1(I, J, K)) =
                                        ((I + i + mx) % mx) +
                                        ((J + j + my) % my) * mx +
                                        ((K + k + mz) % mz) * mx * my;
                                }
                            }
                        }
                    }
                }
            }
        }

        // std::cout << adjOff.transpose() << std::endl;
    }

    std::size_t numCells() const { return mx * my * mz; }

    auto const &getAdjOff() const { return adjOff; }

    double rcut() const { return m_rcut; }

    Lim const &limits(std::size_t i) const { return m_limits[i]; }

    template <typename T> inline std::size_t lambda(Atom<T> const &atom) const {
        return lambda(atom[0], atom[1], atom[2]);
    }

    inline std::size_t lambda(double x, double y, double z) const {
        check(x >= 0 && x < m_limits[0].len, "x out of +cell");
        check(y >= 0 && y < m_limits[1].len, "y out of +cell");
        check(z >= 0 && z < m_limits[2].len, "z out of +cell");

        std::size_t i = x * m_limits[0].inv * mx;
        std::size_t j = y * m_limits[1].inv * my;
        std::size_t k = z * m_limits[2].inv * mz;

        check(map3To1(i, j, k) < numCells(), "lambda out of range");

        return map3To1(i, j, k);
    }

    template <typename K>
    inline double norm(Atom<K> const &a, Atom<K> const &b, double &dx,
                       double &dy, double &dz) {
        dx = b[0] - a[0];
        dy = b[1] - a[1];
        dz = b[2] - a[2];

        check(&a != &b, "atoms in norm are the same atom");

        check(!(b[0] == a[0] && b[1] == a[1] && b[2] == a[2]),
              "two atoms in same position");

        dx -= m_limits[0].len * std::floor(m_limits[0].inv * dx + 0.5);
        dy -= m_limits[1].len * std::floor(m_limits[1].inv * dy + 0.5);
        dz -= m_limits[2].len * std::floor(m_limits[2].inv * dz + 0.5);

        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

// Builds an LCL for a 3N position vector on each call
template <typename C, typename B> class LinkedCellList {
  private:
    using kind_t = typename std::remove_reference_t<C>::value_type;

    std::size_t numAtoms;

    C kinds;
    B box;

    Eigen::Array<std::size_t, Eigen::Dynamic, 1> head;
    std::vector<Atom<kind_t>> list;

  public:
    // kinds is a list of atom species
    // m is number of cells
    // rcut is cut of radius
    // lambda is function-like F : double, double, double -> 0,1,..,m-1
    LinkedCellList(C &&kinds, B &&box)
        : numAtoms{kinds.size()}, kinds{std::forward<C>(kinds)},
          box{std::forward<B>(box)}, head(box.numCells()) {}

    inline auto const &getKinds() const { return kinds; }

    inline auto begin() const { return list.begin(); }
    inline auto end() const { return list.begin() + numAtoms; }

    inline Atom<kind_t> const &getAtom(std::size_t i) const { return list[i]; }
    inline std::size_t getNumAtoms() const { return numAtoms; }

    template <typename T> void makeCellList(T const &x3n) {
        check(static_cast<std::size_t>(x3n.size()) == 3 * numAtoms,
              "wrong number of atoms");

        static_assert(std::is_trivially_destructible_v<Atom<kind_t>>,
                      "can't clear in constant time");

        list.clear(); // should be order 1 by ^

        // add all atoms to cell list
        for (std::size_t i = 0; i < numAtoms; ++i) {
            // map atoms into cell (0->xlen, 0->ylen, 0->zlen)
            double x = x3n[3 * i + 0] -
                       box.limits(0).len *
                           std::floor(x3n[3 * i + 0] * box.limits(0).inv);

            double y = x3n[3 * i + 1] -
                       box.limits(1).len *
                           std::floor(x3n[3 * i + 1] * box.limits(1).inv);

            double z = x3n[3 * i + 2] -
                       box.limits(2).len *
                           std::floor(x3n[3 * i + 2] * box.limits(2).inv);

            std::cout << x << ' ' << y << ' ' << z << ' ' << std::endl;

            list.emplace_back(kinds[i], x, y, z);
        }

        updateHead();

        std::cout << head.transpose() << std::endl;
    }

  private:
    // Alg 3.1
    void updateHead() {
        for (auto &&elem : head) {
            elem = list.size();
        }
        for (std::size_t i = 0; i < list.size(); ++i) {
            // update cell list
            Atom<kind_t> &atom = list[i];

            std::size_t lambda = box.lambda(atom);
            atom.next() = head[lambda];
            head[lambda] = i;
        }
    }

  public:
    // applies f() for every neighbour (closer than rcut)
    template <typename F>
    inline void findNeigh(Atom<kind_t> const &atom, F const &f) {

        std::size_t const lambda = box.lambda(atom);
        std::size_t const end = list.size();

        std::size_t index = head[lambda];

        // in same cell as atom
        do {
            Atom<kind_t> const &neigh = list[index];
            if (&atom != &neigh) {
                double dx, dy, dz;
                double r = box.norm(neigh, atom, dx, dy, dz);
                if (r <= box.rcut()) {
                    f(neigh, r, dx, dy, dz);
                }
            }
            index = neigh.next();
        } while (index != end);

        // in adjecent cells -- don't need check against self
        for (std::size_t cell : box.getAdjOff().col(lambda)) {
            index = head[cell];
            while (index != end) {
                std::cout << "cell_t " << cell << std::endl;
                Atom<kind_t> const &neigh = list[index];
                double dx, dy, dz;
                double r = box.norm(neigh, atom, dx, dy, dz);
                if (r <= box.rcut()) {
                    f(neigh, r, dx, dy, dz);
                }
                index = neigh.next();
            }
        }
    }
};
template <typename C, typename B>
LinkedCellList(C &&, B &&)->LinkedCellList<C, B>;

template <typename C> class CompEAM {
  private:
    TabEAM data;
    LinkedCellList<C, Box> lcl;

  public:
    CompEAM(std::string const &file, C &&kinds, double xmin, double xmax,
            double ymin, double ymax, double zmin, double zmax)
        : data{parseTabEAM(file)}, lcl{std::forward<C>(kinds),
                                       {data.rCut, xmin, xmax, ymin, ymax, zmin,
                                        zmax}} {
        using kind_t = typename std::remove_reference_t<C>::value_type;

        static_assert(std::is_integral_v<kind_t>, "kinds must be integers");

        check(*std::max_element(lcl.getKinds().begin(), lcl.getKinds().end()) <
                  kind_t(data.numSpecies),
              "found kinds that are not in the EAM species data");

        check(*std::min_element(lcl.getKinds().begin(), lcl.getKinds().end()) >=
                  kind_t(0),
              "found kinds that are negative?");
    }

    // compute energy
    template <typename T> double operator()(T const &x) {
        lcl.makeCellList(x);

        double sum = 0;
        for (auto const &alpha : lcl) {
            double rho = 0;

            lcl.findNeigh(alpha, [&](auto const &beta, double r, double, double,
                                     double) {
                sum += 0.5 *
                       data.tabV(alpha.kind(), beta.kind())[data.rToIndex(r)];

                rho += data.tabPhi(beta.kind(), alpha.kind())[data.rToIndex(r)];
            });

            sum += data.tabF.col(alpha.kind())[data.pToIndex(rho)];
        }
        return sum;
    }

    // assumes makeCellList has already been called
    template <typename K> double calcRho(Atom<K> const &beta) {
        double rho = 0;
        lcl.findNeigh(
            beta, [&](auto const &alpha, double r, double, double, double) {
                rho += data.tabPhi(alpha.kind(), beta.kind())[data.rToIndex(r)];
            });
        return rho;
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) {
        lcl.makeCellList(x);

        for (std::size_t i = 0; i < lcl.getNumAtoms(); ++i) {
            auto const &gamma = lcl.getAtom(i);

            out[3 * i + 0] = 0;
            out[3 * i + 1] = 0;
            out[3 * i + 2] = 0;

            double fpg =
                data.difF.col(gamma.kind())[data.pToIndex(calcRho(gamma))];

            // finds R^{\alpha\gamma}
            lcl.findNeigh(gamma, [&](auto const &alpha, double r, double dx,
                                     double dy, double dz) {
                std::cout << "r is: " << r << ' ' << gamma[0] << std::endl;

                double mag =
                    data.difV(alpha.kind(), gamma.kind())[data.rToIndex(r)] +

                    fpg * data.difPhi(alpha.kind(),
                                      gamma.kind())[data.rToIndex(r)] +

                    data.difF.col(alpha.kind())[data.pToIndex(calcRho(alpha))] *

                        data.difPhi(gamma.kind(),
                                    alpha.kind())[data.rToIndex(r)];

                out[3 * i + 0] += dx * mag / r;
                out[3 * i + 1] += dy * mag / r;
                out[3 * i + 2] += dz * mag / r;
            });
        }
    }
};

template <typename C>
CompEAM(std::string const &, C &&, double, double, double, double, double,
        double)
    ->CompEAM<C>;
