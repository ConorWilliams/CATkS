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

    std::array<long, 124> adjOff; // 3^3 - 1

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
        // than rcut, plus two (+ 4) for gost cell layer
        mx = static_cast<std::size_t>((xmax - xmin) / (rcut * 0.5)) + 8;
        my = static_cast<std::size_t>((ymax - ymin) / (rcut * 0.5)) + 8;
        mz = static_cast<std::size_t>((zmax - zmin) / (rcut * 0.5)) + 8;

        std::cout << "shape " << mx << ' ' << my << ' ' << mz << std::endl;

        // dimensions of cells
        double lcx = (xmax - xmin) / (mx - 8);
        double lcy = (ymax - ymin) / (my - 8);
        double lcz = (ymax - ymin) / (mz - 8);

        ox = -4 * lcx; // o for offset
        oy = -4 * lcy;
        oz = -4 * lcz;

        lx = lcx * mx;
        ly = lcy * my;
        lz = lcz * mz;

        // compute neighbour offsets
        long idx = 0;
        for (auto k : {-2, -1, 0, 1, 2}) {
            for (auto j : {-2, -1, 0, 1, 2}) {
                for (auto i : {-2, -1, 0, 1, 2}) {
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
    inline std::array<double, 3> mapIntoCell(double x, double y, double z) {

        std::array<double, 3> ret;

        ret[0] = x - limits(0).len * std::floor(x * limits(0).inv);
        ret[1] = y - limits(1).len * std::floor(y * limits(1).inv);
        ret[2] = z - limits(2).len * std::floor(z * limits(2).inv);

        // ret[0] -= limits(0).len * std::floor(x * limits(0).inv);
        // ret[1] -= limits(1).len * std::floor(y * limits(1).inv);
        // ret[2] -= limits(2).len * std::floor(z * limits(2).inv);

        return ret;
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

    template <typename T> void fillCellList(T const &x3n) {
        check(static_cast<std::size_t>(x3n.size()) == 3 * numAtoms,
              "wrong number of atoms");

        static_assert(std::is_trivially_destructible_v<Atom<kind_t>>,
                      "can't clear atoms in constant time");

        list.clear(); // should be order 1 by ^

        // add all atoms to cell list
        for (std::size_t i = 0; i < numAtoms; ++i) {
            // map atoms into cell (0->xlen, 0->ylen, 0->zlen)
            auto [x, y, z] =
                box.mapIntoCell(x3n[3 * i + 0], x3n[3 * i + 1], x3n[3 * i + 2]);

            // check in cell
            check(x >= 0 && x < box.limits(0).len, "x out of box");
            check(y >= 0 && y < box.limits(1).len, "y out of box");
            check(z >= 0 && z < box.limits(2).len, "z out of box");

            list.emplace_back(kinds[i], x, y, z);
        }
    }

    template <typename T> void makeCellList(T const &x) {
        fillCellList(x);
        makeGhosts();
        updateHead();
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
            // fixed end stops double copies
            std::size_t end = list.size();

            for (std::size_t j = 0; j < end; ++j) {
                Atom<kind_t> atom = list[j];

                if (atom[i] < 2 * box.rcut()) {

                    list.push_back(atom);
                    list.back()[i] += box.limits(i).len;
                }

                if (atom[i] > box.limits(i).len - 2 * box.rcut()) {

                    list.push_back(atom);
                    list.back()[i] -= box.limits(i).len;
                }
            }
        }
    }

  public:
    // applies f() for every neighbour (closer than rcut)
    template <typename F>
    inline void findNeigh(Atom<kind_t> const &atom, F const &f) {

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

    void sort(Eigen::ArrayXd &x) {
        fillCellList(x);
        updateHead();

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
template <typename C, typename B>
LinkedCellList(C &&, B &&)->LinkedCellList<C, B>;

/////////////////////////////////////////////////////////////////////////////////

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
    template <typename K> double calcRho(Atom<K> const &beta) const {
        double rho = 0;
        lcl.findNeigh(
            beta, [&](auto const &alpha, double r, double, double, double) {
                rho += data.ineterpR(data.tabPhi(alpha.kind(), beta.kind()), r);
            });
        return rho;
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) const {

        lcl.makeCellList(x);

        for (std::size_t i = 0; i < lcl.getNumAtoms(); ++i) {
            auto const &gamma = lcl.getAtom(i);

            out[3 * i + 0] = 0;
            out[3 * i + 1] = 0;
            out[3 * i + 2] = 0;

            double const fpg =
                data.ineterpP(data.difF.col(gamma.kind()), calcRho(gamma));

            // finds R^{\alpha\gamma}
            lcl.findNeigh(gamma, [&](auto const &alpha, double r, double dx,
                                     double dy, double dz) {
                // std::cout << "r is: " << r << ' ' << gamma[0] << std::endl;

                double mag =
                    data.ineterpR(data.difV(alpha.kind(), gamma.kind()), r) +
                    fpg * data.ineterpR(data.difPhi(alpha.kind(), gamma.kind()),
                                        r) +
                    data.ineterpP(data.difF.col(alpha.kind()), calcRho(alpha)) *
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
