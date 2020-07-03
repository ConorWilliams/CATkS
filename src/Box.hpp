#pragma once

#include <array>
#include <cmath>

#include "utils.hpp"

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
        CHECK(xmax > xmin, "cannot have x min <= max");
        CHECK(ymax > ymin, "cannot have y min <= max");
        CHECK(zmax > zmin, "cannot have z min <= max");

        CHECK(rcut > 0, "rcut is negative");

        CHECK(xmax - xmin >= rcut, "box too small");
        CHECK(ymax - ymin >= rcut, "box too small");
        CHECK(zmax - zmin >= rcut, "box too small");

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

        CHECK(lcx >= rcut, "cells made too small");
        CHECK(lcy >= rcut, "cells made too small");
        CHECK(lcz >= rcut, "cells made too small");

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

    std::array<long, 26> const &getAdjOff() const { return adjOff; }

    double rcut() const { return m_rcut; }

    Lim const &limits(std::size_t i) const { return m_limits[i]; }

    template <typename T>
    inline double normSq(T const &a, T const &b, double &dx, double &dy,
                         double &dz) const {
        dx = b[0] - a[0];
        dy = b[1] - a[1];
        dz = b[2] - a[2];

        CHECK(&a != &b, "atoms in norm are the same atom");

        CHECK(!(b[0] == a[0] && b[1] == a[1] && b[2] == a[2]),
              "two atoms in same position");

        return dx * dx + dy * dy + dz * dz;
    }

    inline Eigen::Vector3d minImage(Eigen::Vector3d dr) const {
        return {
            dr[0] - limits(0).len * std::floor(dr[0] * limits(0).inv + 0.5),
            dr[1] - limits(1).len * std::floor(dr[1] * limits(1).inv + 0.5),
            dr[2] - limits(2).len * std::floor(dr[2] * limits(2).inv + 0.5),
        };
    }

    // inline double periodicNormSq(double x1, double y1, double z1, double x2,
    //                              double y2, double z2) const {
    //     // double dx = std::abs(x1 - x2);
    //     // double dy = std::abs(y1 - y2);
    //     // double dz = std::abs(z1 - z2);
    //     //
    //     // dx -= static_cast<int>(dx * limits(0).inv + 0.5) * limits(0).len;
    //     // dy -= static_cast<int>(dy * limits(1).inv + 0.5) * limits(1).len;
    //     // dz -= static_cast<int>(dz * limits(2).inv + 0.5) * limits(2).len;
    //
    //     double dx = x1 - x2;
    //     double dy = y1 - y2;
    //     double dz = z1 - z2;
    //
    //     dx -= limits(0).len * std::floor(dx * limits(0).inv + 0.5);
    //     dy -= limits(1).len * std::floor(dy * limits(1).inv + 0.5);
    //     dz -= limits(2).len * std::floor(dz * limits(2).inv + 0.5);
    //
    //     return dx * dx + dy * dy + dz * dz;
    // }

    template <typename T> inline std::size_t lambda(T const &atom) const {
        CHECK(atom[0] >= ox && atom[0] - ox < lx, "x out of +cell");
        CHECK(atom[1] >= oy && atom[1] - oy < ly, "y out of +cell");
        CHECK(atom[2] >= oz && atom[2] - oz < lz, "z out of +cell");

        std::size_t i = ((atom[0] - ox) * mx) / lx;
        std::size_t j = ((atom[1] - oy) * my) / ly;
        std::size_t k = ((atom[2] - oz) * mz) / lz;

        CHECK(i + j * mx + k * mx * my < numCells(), "lambda out of range");

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
