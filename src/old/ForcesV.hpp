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
    std::array<double, 3> r;
    double m_rho;
    T k;
    std::size_t n;

  public:
    Atom(T k, double x, double y, double z) : r{x, y, z}, m_rho{0.0}, k{k} {}
    Atom(Atom const &) = default;
    Atom(Atom &&) = default;
    Atom() = default;
    Atom &operator=(Atom const &) = default;
    Atom &operator=(Atom &&) = default;

    inline T const &kind() const { return k; }

    inline std::size_t &next() { return n; }
    inline std::size_t next() const { return n; }

    inline double &rho() { return m_rho; }
    inline double rho() const { return m_rho; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

// Contains info about simulation box and performs Lambda mapping
class Box {
  private:
    double m_rcut;

    struct Lim {
        double min, max, len, inv;
        Lim(double min, double max)
            : min{min}, max{max}, len{max - min}, inv{1.0 / len} {}
    };

    std::array<Lim, 3> m_limits;

    double ox; // offsets
    double oy;
    double oz;

    std::size_t mx; // number of cells
    std::size_t my;
    std::size_t mz;

    double lx; // lengths
    double ly;
    double lz;

    std::array<long, 26> adjOff; // 3^3 - 1

  public:
    Box(double rcut, double xmin, double xmax, double ymin, double ymax,
        double zmin, double zmax)
        : m_rcut{rcut}, m_limits{{{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}} {
        // sanity
        check(xmax > xmin, "cannot have x min <= max");
        check(ymax > ymin, "cannot have y min <= max");
        check(zmax > zmin, "cannot have z min <= max");

        check(rcut > 0, "rcut is negative");

        check(xmax - xmin >= rcut, "box too small");
        check(ymax - ymin >= rcut, "box too small");
        check(zmax - zmin >= rcut, "box too small");

        // round down (toward zero) truncation, therefore boxes always biger
        // than rcut, plus two (+ 2) for gost cell layer
        mx = static_cast<std::size_t>((xmax - xmin) / rcut) + 2;
        my = static_cast<std::size_t>((ymax - ymin) / rcut) + 2;
        mz = static_cast<std::size_t>((zmax - zmin) / rcut) + 2;

        std::cout << "shape " << mx << ' ' << my << ' ' << mz << std::endl;

        // dimensions of cells
        double lcx = (xmax - xmin) / (mx - 2);
        double lcy = (ymax - ymin) / (my - 2);
        double lcz = (ymax - ymin) / (mz - 2);

        check(lcx >= rcut, "cells made too small");
        check(lcy >= rcut, "cells made too small");
        check(lcz >= rcut, "cells made too small");

        ox = -lcx; // o for offset
        oy = -lcy;
        oz = -lcz;

        lx = lcx * mx;
        ly = lcy * my;
        lz = lcz * mz;

        // compute neighbour offsets
        long idx = 0;
        for (auto k : {-1, 0, 1}) {
            for (auto j : {-1, 0, 1}) {
                for (auto i : {-1, 0, 1}) {
                    if (i != 0 || j != 0 || k != 0) {
                        adjOff[idx++] = i + j * mx + k * mx * my;
                    }
                }
            }
        }

        for (auto n : adjOff) {
            std::cout << n << ' ';
        }
        std::cout << std::endl;
    }

    std::size_t numCells() const { return mx * my * mz; }

    auto const &getAdjOff() const { return adjOff; }

    double rcut() const { return m_rcut; }

    Lim const &limits(std::size_t i) const { return m_limits[i]; }

    template <typename K>
    inline double norm(Atom<K> const &a, Atom<K> const &b, double &dx,
                       double &dy, double &dz) const {
        dx = b[0] - a[0];
        dy = b[1] - a[1];
        dz = b[2] - a[2];

        check(&a != &b, "atoms in norm are the same atom");

        check(!(b[0] == a[0] && b[1] == a[1] && b[2] == a[2]),
              "two atoms in same position");

        return dx * dx + dy * dy + dz * dz;
    }

    template <typename T> inline std::size_t lambda(Atom<T> const &atom) const {
        check(atom[0] >= ox && atom[0] - ox < lx, "x out of +cell");
        check(atom[1] >= oy && atom[1] - oy < ly, "y out of +cell");
        check(atom[2] >= oz && atom[2] - oz < lz, "z out of +cell");

        std::size_t i = (atom[0] - ox) * mx / lx;
        std::size_t j = (atom[1] - oy) * my / ly;
        std::size_t k = (atom[2] - oz) * mz / lz;

        check(i + j * mx + k * mx * my < numCells(), "lambda out of range");

        return i + j * mx + k * mx * my;
    }

    // maps point any were into range 0->(max-min)
    inline std::array<double, 3> mapIntoCell(double x, double y,
                                             double z) const {

        std::array<double, 3> ret;

        ret[0] = x - limits(0).len * std::floor(x * limits(0).inv);
        ret[1] = y - limits(1).len * std::floor(y * limits(1).inv);
        ret[2] = z - limits(2).len * std::floor(z * limits(2).inv);

        // ret[0] -= limits(0).len * std::floor(ret[0] * limits(0).inv);
        // ret[1] -= limits(1).len * std::floor(ret[1] * limits(1).inv);
        // ret[2] -= limits(2).len * std::floor(ret[2] * limits(2).inv);
        // else
        ret[0] = ret[0] == limits(0).len ? 0.0 : ret[0];
        ret[1] = ret[1] == limits(1).len ? 0.0 : ret[1];
        ret[2] = ret[2] == limits(2).len ? 0.0 : ret[2];

        return ret;
    }
};

template <typename C> class FuncEAM {
  private:
    using kind_t = typename std::remove_reference_t<C>::value_type;

    TabEAM data;
    C kinds;
    Box box;

    std::size_t numAtoms;

    mutable Eigen::Array<std::size_t, Eigen::Dynamic, 1> head;
    mutable std::vector<Atom<kind_t>> list;

    mutable std::vector<std::vector<Atom<kind_t> *>> verlet;

  public:
    FuncEAM(std::string const &file, C &&kinds, double xmin, double xmax,
            double ymin, double ymax, double zmin, double zmax)
        : data{parseTabEAM(file)}, kinds{std::forward<C>(kinds)},
          box{data.rCut, xmin, xmax, ymin, ymax, zmin, zmax},
          numAtoms{kinds.size()}, head(box.numCells()), verlet(numAtoms) {

        static_assert(std::is_integral_v<kind_t>, "kinds must be integers");

        check(*std::max_element(kinds.begin(), kinds.end()) <
                  kind_t(data.numSpecies),
              "found kinds that are not in the EAM species data");

        check(*std::min_element(kinds.begin(), kinds.end()) >= kind_t(0),
              "found kinds that are negative?");

        static_assert(std::is_trivially_destructible_v<Atom<kind_t>>,
                      "can't clear atoms in constant time");
    }

  private:
    template <typename T> inline void fillCellList(T const &x3n) const {
        check(static_cast<std::size_t>(x3n.size()) == 3 * numAtoms,
              "wrong number of atoms");

        list.clear(); // should be order 1

        // add all atoms to cell list
        for (std::size_t i = 0; i < numAtoms; ++i) {
            // map atoms into cell (0->xlen, 0->ylen, 0->zlen)
            auto [x, y, z] =
                box.mapIntoCell(x3n[3 * i + 0], x3n[3 * i + 1], x3n[3 * i + 2]);

            // check in cell
            check(x >= 0 && x < box.limits(0).len, "x out of box " << x);
            check(y >= 0 && y < box.limits(1).len, "y out of box " << y);
            check(z >= 0 && z < box.limits(2).len, "z out of box " << z);

            list.emplace_back(kinds[i], x, y, z);
        }
    }

    // Alg 3.1
    inline void updateHead() const {
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

    // Rapaport p.18
    inline void makeGhosts() const {
        // TODO: could accelerate by only looking for ghosts at edge boxes
        for (std::size_t i = 0; i < 3; ++i) {
            // fixed end stops double copies
            std::size_t end = list.size();

            for (std::size_t j = 0; j < end; ++j) {
                Atom<kind_t> atom = list[j];

                if (atom[i] < box.rcut()) {
                    list.push_back(atom);
                    list.back()[i] += box.limits(i).len;
                }

                if (atom[i] > box.limits(i).len - box.rcut()) {
                    list.push_back(atom);
                    list.back()[i] -= box.limits(i).len;
                }
            }
        }
    }

    // applies f() for every neighbour (closer than rcut)
    template <typename F>
    __attribute__((noinline)) inline void findNeigh(Atom<kind_t> const &atom,
                                                    F const &f) const {

        std::size_t const lambda = box.lambda(atom);
        std::size_t const end = list.size();
        double const cut_sq = box.rcut() * box.rcut();

        std::size_t index = head[lambda];

        // in same cell as atom
        do {
            Atom<kind_t> const &neigh = list[index];
            if (&atom != &neigh) {
                double dx, dy, dz;
                double r_sq = box.norm(neigh, atom, dx, dy, dz);
                if (r_sq <= cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }
            }
            index = neigh.next();
        } while (index != end);

        // in adjecent cells -- don't need check against self
        for (std::size_t off : box.getAdjOff()) {
            index = head[lambda + off];
            while (index != end) {
                // std::cout << "cell_t " << cell << std::endl;
                Atom<kind_t> const &neigh = list[index];
                double dx, dy, dz;
                double r_sq = box.norm(neigh, atom, dx, dy, dz);
                if (r_sq <= cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }
                index = neigh.next();
            }
        }
    }

    void makeVerlet() {
        for (std::size_t i = 0; i < numAtoms; ++i) {
            verlet[i].clear();
            Atom<kind_t> *atom = &list[i];

            std::size_t const lambda = box.lambda(*atom);
            std::size_t const end = list.size();
            double const cut_sq = box.rcut() * box.rcut();

            std::size_t index = head[lambda];

            // in same cell as atom
            do {
                Atom<kind_t> *neigh = &list[index];
                if (atom != neigh) {
                    double dx, dy, dz;
                    double r_sq = box.norm(*neigh, *atom, dx, dy, dz);
                    if (r_sq <= cut_sq) {
                        verlet[i].push_back(neigh);
                    }
                }
                index = neigh->next();
            } while (index != end);

            // in adjecent cells -- don't need check against self
            for (std::size_t off : box.getAdjOff()) {
                index = head[lambda + off];
                while (index != end) {
                    // std::cout << "cell_t " << cell << std::endl;
                    Atom<kind_t> *neigh = &list[index];
                    double dx, dy, dz;
                    double r_sq = box.norm(*neigh, *atom, dx, dy, dz);
                    if (r_sq <= cut_sq) {
                        verlet[i].push_back(neigh);
                    }
                    index = neigh->next();
                }
            }
        }
    };

  public:
    // compute energy
    template <typename T> double operator()(T const &x) const {

        fillCellList(x);
        makeGhosts();
        updateHead();

        double sum = 0;
        for (auto it = list.begin(); it != list.begin() + numAtoms; ++it) {

            findNeigh(*it, [&](auto const &beta, double r, double, double,
                               double) {
                sum +=
                    0.5 * data.ineterpR(data.tabV(it->kind(), beta.kind()), r);

                it->rho() +=
                    data.ineterpR(data.tabPhi(beta.kind(), it->kind()), r);
            });

            sum += data.ineterpP(data.tabF.col(it->kind()), it->rho());
        }
        return sum;
    }

    // assumes makeCellList has already been called
    void __attribute__((noinline)) calcAllRho() const {
        for (auto it = list.begin(); it != list.begin() + numAtoms; ++it) {
            findNeigh(
                *it, [&](auto const &alpha, double r, double, double, double) {
                    it->rho() +=
                        data.ineterpR(data.tabPhi(alpha.kind(), it->kind()), r);
                });
        }

        list.resize(numAtoms); // should not realloc, verlet still valid
        makeGhosts();
        updateHead();
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) const {

        fillCellList(x);
        makeGhosts();
        updateHead();

        calcAllRho();

        for (std::size_t i = 0; i < numAtoms; ++i) {
            auto const &gamma = list[i];

            out[3 * i + 0] = 0;
            out[3 * i + 1] = 0;
            out[3 * i + 2] = 0;

            double const fpg =
                data.ineterpP(data.difF.col(gamma.kind()), gamma.rho());

            // finds R^{\alpha\gamma}
            findNeigh(gamma, [&](auto const &alpha, double r, double dx,
                                 double dy, double dz) {
                // std::cout << "r is: " << r << ' ' << gamma[0] << std::endl;

                double mag =
                    data.ineterpR(data.difV(alpha.kind(), gamma.kind()), r) +
                    fpg * data.ineterpR(data.difPhi(alpha.kind(), gamma.kind()),
                                        r) +
                    data.ineterpP(data.difF.col(alpha.kind()), alpha.rho()) *
                        data.ineterpR(data.difPhi(gamma.kind(), alpha.kind()),
                                      r);

                out[3 * i + 0] += dx * mag / r;
                out[3 * i + 1] += dy * mag / r;
                out[3 * i + 2] += dz * mag / r;
            });
        }
    }

    // remaps into cell and performs sort
    void sort(Eigen::ArrayXd &x) const {

        fillCellList(x);

        std::sort(list.begin(), list.end(),
                  [&](Atom<kind_t> const &a, Atom<kind_t> const &b) -> bool {
                      return box.lambda(a) < box.lambda(b);
                  });

        for (std::size_t i = 0; i < list.size(); ++i) {
            x[3 * i + 0] = list[i][0];
            x[3 * i + 1] = list[i][1];
            x[3 * i + 2] = list[i][2];

            kinds[i] = list[i].kind();
        }
    }
};

template <typename C>
FuncEAM(std::string const &, C &&, double, double, double, double, double,
        double)
    ->FuncEAM<C>;
