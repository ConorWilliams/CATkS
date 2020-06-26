#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "Box.hpp"
#include "EAM.hpp"
#include "Rdf.hpp"
#include "sort.hpp"
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

template <typename C> class FuncEAM {
  private:
    using kind_t = typename std::remove_reference_t<C>::value_type;

    TabEAM data;
    C kinds;
    Box box;

    std::size_t numAtoms;

    mutable Eigen::Array<std::size_t, Eigen::Dynamic, 1> head;
    mutable std::vector<Atom<kind_t>> list;

  public:
    FuncEAM(std::string const &file, C &&kinds, double xmin, double xmax,
            double ymin, double ymax, double zmin, double zmax)
        : data{parseTabEAM(file)}, kinds{std::forward<C>(kinds)},
          box{data.rCut, xmin, xmax, ymin, ymax, zmin, zmax},
          numAtoms{kinds.size()}, head(box.numCells()) {

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
    template <typename T> void fillCellList(T const &x3n) const {
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
    void updateHead() const {
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
    void makeGhosts() const {
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
    inline void findNeigh(Atom<kind_t> const &atom, F const &f) const {

        std::size_t const lambda = box.lambda(atom);
        std::size_t const end = list.size();
        double const cut_sq = box.rcut() * box.rcut();

        std::size_t index = head[lambda];

        // in same cell as atom
        do {
            Atom<kind_t> const &neigh = list[index];
            if (&atom != &neigh) {
                double dx, dy, dz;
                double r_sq = box.normSq(neigh, atom, dx, dy, dz);
                if (r_sq <= cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }
            }
            index = neigh.next();
        } while (index != end);

        // in adjecent cells -- don't need check against self
        for (auto off : box.getAdjOff()) {
            index = head[lambda + off];
            while (index != end) {
                // std::cout << "cell_t " << cell << std::endl;
                Atom<kind_t> const &neigh = list[index];
                double dx, dy, dz;
                double r_sq = box.normSq(neigh, atom, dx, dy, dz);
                if (r_sq <= cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }
                index = neigh.next();
            }
        }
    }

  public:
    // compute energy
    template <typename T> double operator()(T const &x) const {

        fillCellList(x); // initilises rho to zeros
        makeGhosts();
        updateHead();

        double sum = 0;
        for (auto alpha = list.begin(); alpha != list.begin() + numAtoms;
             ++alpha) {

            findNeigh(*alpha, [&](auto const &beta, double r, double, double,
                                  double) {
                sum += 0.5 *
                       data.ineterpR(data.tabV(alpha->kind(), beta.kind()), r);

                alpha->rho() +=
                    data.ineterpR(data.tabPhi(beta.kind(), alpha->kind()), r);
            });

            sum += data.ineterpP(data.tabF.col(alpha->kind()), alpha->rho());
        }
        return sum;
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) const {

        fillCellList(x);
        makeGhosts();
        updateHead();

        // computes all rho values
        for (auto beta = list.begin(); beta != list.begin() + numAtoms;
             ++beta) {
            findNeigh(*beta, [&](auto const &alpha, double r, double, double,
                                 double) {
                beta->rho() +=
                    data.ineterpR(data.tabPhi(alpha.kind(), beta->kind()), r);
            });
        }

        list.resize(numAtoms); // deletes ghosts, should not realloc
        makeGhosts();          // ghosts now have correct rho values
        updateHead();

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
                double mag =
                    (data.ineterpR(data.difV(alpha.kind(), gamma.kind()), r) +
                     fpg * data.ineterpR(
                               data.difPhi(alpha.kind(), gamma.kind()), r) +
                     data.ineterpP(data.difF.col(alpha.kind()), alpha.rho()) *
                         data.ineterpR(data.difPhi(gamma.kind(), alpha.kind()),
                                       r)) /
                    r;

                out[3 * i + 0] += dx * mag;
                out[3 * i + 1] += dy * mag;
                out[3 * i + 2] += dz * mag;
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
        // else
        // cj::sort_clever(list.begin(), list.end(), 0, box.numCells(),
        //                 [&](auto const &atom) { return box.lambda(atom); });

        for (std::size_t i = 0; i < list.size(); ++i) {
            x[3 * i + 0] = list[i][0];
            x[3 * i + 1] = list[i][1];
            x[3 * i + 2] = list[i][2];

            kinds[i] = list[i].kind();
        }
    }

    auto rcut() const { return box.rcut(); }

    auto getBox() const { return box; }

    template <typename T> auto colourAll(T const &x) const {
        fillCellList(x);
        makeGhosts();
        updateHead();

        std::vector<Rdf> colours;

        for (auto atom = list.begin(); atom != list.begin() + numAtoms;
             ++atom) {

            Rdf rdf{};

            findNeigh(*atom, [&](auto const &, double r, double, double,
                                 double) { rdf.add(r / box.rcut()); });

            colours.push_back(rdf);
        }

        return colours;
    }

    template <typename T> auto quasiColourAll(T const &x) const {
        fillCellList(x);
        makeGhosts();
        updateHead();

        std::vector<std::size_t> colours;

        for (auto atom = list.begin(); atom != list.begin() + numAtoms;
             ++atom) {

            std::size_t count = 0;

            findNeigh(*atom,
                      [&](auto const &, double r, double, double, double) {
                          if (r < 2.6) {
                              ++count;
                          }
                      });

            colours.push_back(count);
        }

        return colours;
    }

    inline Eigen::Vector3d minImage(Eigen::Vector3d dr) const {
        return box.minImage(dr[0], dr[1], dr[2]);
    }

    inline double periodicNormSq(double x1, double y1, double z1, double x2,
                                 double y2, double z2) const {

        return box.minImage(x2 - x1, y2 - y1, z2 - z1).squaredNorm();
    }
};

template <typename C>
FuncEAM(std::string const &, C &&, double, double, double, double, double,
        double)
    ->FuncEAM<C>;
