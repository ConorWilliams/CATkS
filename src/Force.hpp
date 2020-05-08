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
    double m_rcut;

    struct Lim {
        double min, max, len;
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
        : m_rcut{rcut}, m_limits{{{xmin, xmax, xmax - xmin},
                                  {ymin, ymax, ymax - ymin},
                                  {zmin, zmax, zmax - zmin}}} {
        // sanity
        check(xmax > xmin, "cannot have x min <= max");
        check(ymax > ymin, "cannot have y min <= max");
        check(zmax > zmin, "cannot have z min <= max");
        check(rcut > 0, "rcut is negative");

        // round down (toward zero) truncation, therefore boxes always biger
        // than rcut, plus two (+ 2) for gost cell layer
        mx = static_cast<std::size_t>((xmax - xmin) / rcut) + 2;
        my = static_cast<std::size_t>((ymax - ymin) / rcut) + 2;
        mz = static_cast<std::size_t>((zmax - zmin) / rcut) + 2;

        // dimensions of cells
        double lcx = (xmax - xmin) / (mx - 2);
        double lcy = (ymax - ymin) / (my - 2);
        double lcz = (ymax - ymin) / (mz - 2);

        check(lcx >= rcut, "cells made too small");
        check(lcy >= rcut, "cells made too small");
        check(lcz >= rcut, "cells made too small");

        ox = -lcx;
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

    template <typename T> inline std::size_t lambda(Atom<T> const &atom) const {
        return lambda(atom[0], atom[1], atom[2]);
    }

    inline std::size_t lambda(double x, double y, double z) const {
        check(x >= ox && x - ox < lx, "x out of +cell");
        check(y >= oy && y - oy < ly, "y out of +cell");
        check(z >= oz && z - oz < lz, "z out of +cell");

        std::size_t i = (x - ox) * mx / lx;
        std::size_t j = (y - oy) * my / ly;
        std::size_t k = (z - oz) * mz / lz;

        check(i + j * mx + k * mx * my < numCells(), "lambda out of range");

        return i + j * mx + k * mx * my;
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

    template <typename T> void makeCellList(T const &x3n) {
        check(static_cast<std::size_t>(x3n.size()) == 3 * numAtoms,
              "wrong number of atoms");

        static_assert(std::is_trivially_destructible_v<Atom<kind_t>>,
                      "can't clear in constant time");

        list.clear(); // should be order 1 by ^

        // add all atoms to cell list
        for (std::size_t i = 0; i < numAtoms; ++i) {
            // map atoms into cell (0->xlen, 0->ylen, 0->zlen)
            double x = std::fmod(x3n[3 * i + 0] - box.limits(0).min,
                                 box.limits(0).len);
            double y = std::fmod(x3n[3 * i + 1] - box.limits(1).min,
                                 box.limits(1).len);
            double z = std::fmod(x3n[3 * i + 2] - box.limits(2).min,
                                 box.limits(2).len);
            // check in cell
            check(x >= 0 && x < box.limits(0).len, "x out of box");
            check(y >= 0 && y < box.limits(1).len, "y out of box");
            check(z >= 0 && z < box.limits(2).len, "z out of box");

            list.emplace_back(kinds[i], x, y, z);
        }

        makeGhosts();

        updateHead();

        for (auto &&atom : list) {
            std::cout << atom[0] << ' ' << atom[1] << ' ' << atom[2] << ' '
                      << std::endl;
        }
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

    // Rapaport p.18
    void makeGhosts() {
        // TODO: could accelerate by only looking for ghosts at edge boxes
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0, end = list.size(); j < end; ++j) {
                Atom<kind_t> const &atom = list[j];

                if (atom[i] < box.rcut()) {
                    list.emplace_back(atom);
                    list.back()[i] += box.limits(i).len;
                } else if (atom[i] > box.limits(i).len - box.rcut()) {
                    list.emplace_back(atom);
                    list.back()[i] -= box.limits(i).len;
                }
            }
        }
    }

  public:
    // applies f() for every neighbour (closer than rcut)
    template <typename F>
    inline void findAdjAtoms(Atom<kind_t> const &atom, F const &f) {

        std::size_t const lambda = box.lambda(atom);
        std::size_t const end = list.size();

        std::size_t index = head[lambda];

        // in same cell as atom
        do {
            Atom<kind_t> const &neigh = list[index];
            if (&atom != &neigh) {
                f(neigh);
            }
            index = neigh.next();
        } while (index != end);

        // in adjecent cells -- don't need check against self
        for (std::size_t offset : box.getAdjOff()) {
            index = head[lambda + offset];
            while (index != end) {
                Atom<kind_t> const &neigh = list[index];
                f(neigh);
                index = neigh.next();
            }
        }
    }
};
template <typename C, typename B>
LinkedCellList(C &&, B &&)->LinkedCellList<C, B>;

template <typename C> class CompEAM {
  private:
    double rcut;

    TabEAM data;
    LinkedCellList<C, Box> lcl;

  public:
    CompEAM(std::string const &file, C &&kinds, Box box)
        : rcut{box.rcut()}, data{parseTabEAM(file)}, lcl{std::forward<C>(kinds),
                                                         std::move(box)} {
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
            lcl.findAdjAtoms(alpha, [&, rcut = rcut](auto beta) {
                double dx = beta[0] - alpha[0];
                double dy = beta[1] - alpha[1];
                double dz = beta[2] - alpha[2];
                double r = dx * dx + dy * dy + dz * dz;

                if (r <= rcut * rcut) {
                    sum += data.tabV(alpha.kind(), ) std::sqrt(r);
                }
            });
        }

        std::cout << "" << sum << std::endl;

        return 3;
    }
};
template <typename C> CompEAM(std::string const &, C &&, Box)->CompEAM<C>;
