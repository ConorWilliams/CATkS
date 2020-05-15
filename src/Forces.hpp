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
    Atom(T k, double x, double y, double z) : r{x, y, z}, k{k} {}
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
                       double &dy, double &dz) {
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
    __attribute__((noinline)) inline std::array<double, 3>
    mapIntoCell(double x, double y, double z) {

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
    TabEAM data;
    mutable LinkedCellList<C, Box> lcl;

  public:
    FuncEAM(std::string const &file, C &&kinds, double xmin, double xmax,
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

    void sort(Eigen::ArrayXd &x) { lcl.sort(x); }

    // compute energy
    template <typename T> double operator()(T const &x) const {

        lcl.makeCellList(x);

        double sum = 0;
        for (auto const &alpha : lcl) {
            double rho = 0;

            lcl.findNeigh(alpha, [&](auto const &beta, double r, double, double,
                                     double) {
                sum += 0.5 *
                       data.ineterpR(data.tabV(alpha.kind(), beta.kind()), r);

                rho += data.ineterpR(data.tabPhi(beta.kind(), alpha.kind()), r);
            });

            sum += data.ineterpP(data.tabF.col(alpha.kind()), rho);
        }
        return sum;
    }

    // assumes makeCellList has already been called
    void __attribute__((noinline)) calcAllRho() const {
        for (auto &&beta : lcl) {
            beta.rho() = 0;
            lcl.findNeigh(beta, [&](auto const &alpha, double r, double, double,
                                    double) {
                beta.rho() +=
                    data.ineterpR(data.tabPhi(alpha.kind(), beta.kind()), r);
            });
        }
        lcl.updateGhosts();
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) const {

        lcl.makeCellList(x);
        calcAllRho();

        for (std::size_t i = 0; i < lcl.getNumAtoms(); ++i) {
            auto const &gamma = lcl.getAtom(i);

            out[3 * i + 0] = 0;
            out[3 * i + 1] = 0;
            out[3 * i + 2] = 0;

            double const fpg =
                data.ineterpP(data.difF.col(gamma.kind()), gamma.rho());

            // finds R^{\alpha\gamma}
            lcl.findNeigh(gamma, [&](auto const &alpha, double r, double dx,
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
};

template <typename C>
FuncEAM(std::string const &, C &&, double, double, double, double, double,
        double)
    ->FuncEAM<C>;
