#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

#include "EAM.hpp"
#include "utils.hpp"

// Contains info about simulation box and performs Lambda mapping
class Box {
  private:
    double m_rcut;

    struct Lim {
        double min, max;
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

  public:
    Box(double rcut, double xmin, double xmax, double ymin, double ymax,
        double zmin, double zmax)
        : m_rcut{rcut}, m_limits{{{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}} {
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

        ox = xmin - lcx;
        oy = ymin - lcy;
        oz = zmin - lcz;

        lx = lcx * mx;
        ly = lcy * my;
        lz = lcz * mz;
    }

    std::size_t numCells() const { return mx * my * mz; }

    double rcut() const { return m_rcut; }

    Lim const &limits(std::size_t i) const { return m_limits[i]; }

    inline std::size_t lambda(double x, double y, double z) {
        check(x >= ox && x - ox < lx, "x out of cell");
        check(y >= oy && y - oy < ly, "y out of cell");
        check(z >= oz && z - oz < lz, "z out of cell");

        std::size_t i = (x - ox) * mx / lx;
        std::size_t j = (y - oy) * my / ly;
        std::size_t k = (z - oz) * mz / lz;

        check(i + j * mx + k * mx * my < numCells(), "lambda out of range");

        return i + j * mx + k * mx * my;
    }
};

template <typename T> class atom {
  private:
    T k;
    std::array<double, 3> r;
    std::size_t n;

  public:
    atom(T k, double x, double y, double z) : k{k}, r{x, y, z} {}

    inline T const &kind() const { return k; }
    inline std::size_t &next() { return n; }

    // read only access
    inline double x() const { return r[0]; }
    inline double y() const { return r[1]; }
    inline double z() const { return r[2]; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

template <typename C> class LinkedCellList {
  private:
    C const &kinds;
    Box box;

    // reusable temporary for expressions
    Eigen::Array<double, Eigen::Dynamic, 1> tmp_x3n;
    Eigen::Array<std::size_t, Eigen::Dynamic, 1> head;

    using kind_t = typename C::value_type;

    std::vector<atom<kind_t>> list;

  public:
    // kinds is a list of atom species
    // m is number of cells
    // rcut is cut of radius
    // lambda is function-like F : double, double, double -> 0,1,..,m-1
    LinkedCellList(C const &kinds, Box box)
        : kinds{kinds}, box{std::move(box)} {
        head.resize(box.numCells());
    }

    template <typename T> void makeCellList(T const &x3n) {
        check(static_cast<std::size_t>(x3n.size()) == 3 * kinds.size(),
              "wrong number of atoms");

        static_assert(std::is_trivially_destructible_v<atom<kind_t>>,
                      "can't clear in constant time");

        list.clear(); // should be order 1 by ^

        // add all atoms to cell list
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            std::size_t o = 3 * i;
            list.emplace_back(kinds[i], x3n[o + 0], x3n[o + 1], x3n[o + 2]);
        }

        makeGhosts();

        updateHead();

        for (std::size_t i = 0; i < list.size(); ++i) {
            auto &atm = list[i];
            std::cout << i << " " << atm.x() << " " << atm.y() << " " << atm.z()
                      << " " << atm.next() << std::endl;
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
            atom<kind_t> &atm = list[i];

            std::size_t lambda = box.lambda(atm.x(), atm.y(), atm.z());
            atm.next() = head[lambda];
            head[lambda] = i;
        }
    }

    // Rapaport p.18
    void makeGhosts() {
        // TODO: could accelerate by only looking for ghosts at edge boxes
        for (std::size_t i = 0; i < 3; ++i) {
            std::size_t end = list.size();
            for (std::size_t j = 0; j < end; ++j) {
                atom<kind_t> const &atm = list[j];

                if (atm[i] < box.limits(i).min + box.rcut()) {
                    list.emplace_back(atm);
                    list.back()[i] += box.limits(i).max - box.limits(i).min;
                } else if (atm[i] > box.limits(i).max - box.rcut()) {
                    list.emplace_back(atm);
                    list.back()[i] -= box.limits(i).max - box.limits(i).min;
                }
            }
        }
    }
};
