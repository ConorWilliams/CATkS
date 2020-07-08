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
#include "Cell.hpp"
#include "DumpXYX.hpp"
#include "EAM.hpp"
#include "sort.hpp"
#include "utils.hpp"

#include "Canon.hpp"

class FuncEAM {
  private:
    // Helper class holds atom in LIST array of LCL with index of neighbours
    class Atom : public AtomBase {
      private:
        double m_rho;

      public:
        Atom(int k, double x, double y, double z, std::size_t)
            : AtomBase::AtomBase{k, x, y, z}, m_rho{0.0} {}

        Atom() = default;

        inline double &rho() { return m_rho; }
        inline double rho() const { return m_rho; }
    };

    mutable CellList<Atom> cellList;

    TabEAM const &data;

    std::size_t numAtoms;

  public:
    FuncEAM(Box const &box, std::vector<int> const &kinds, TabEAM const &data)
        : cellList{box, kinds}, data{data}, numAtoms{kinds.size()} {

        CHECK(*std::max_element(kinds.begin(), kinds.end()) <
                  int(data.numSpecies()),
              "found kinds that are not in the EAM species data");

        CHECK(*std::min_element(kinds.begin(), kinds.end()) >= int(0),
              "found kinds that are negative?");
    }

  public:
    // compute energy
    template <typename T> double operator()(T const &x) const {

        cellList.fillList(x); // initilises rho to zeros
        cellList.makeGhosts();
        cellList.updateHead();

        // Vector to_out{x.size()};
        //
        // for (int i = 0; i < x.size(); ++i) {
        //     to_out[i] = cellList[i / 3][i % 3];
        // }
        //
        // output(to_out);

        double sum = 0;

        for (auto &&alpha : cellList) {

            cellList.forEachNeigh(
                alpha, [&](auto const &beta, double r, double, double, double) {
                    sum += 0.5 * data.getV(alpha.kind(), beta.kind())(r);

                    alpha.rho() += data.getP(beta.kind(), alpha.kind())(r);
                });

            sum += data.getF(alpha.kind())(alpha.rho());
        }
        return sum;
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) const {

        cellList.fillList(x);
        cellList.makeGhosts();
        cellList.updateHead();

        // computes all rho values
        for (auto &&beta : cellList) {
            cellList.forEachNeigh(
                beta, [&](auto const &alpha, double r, double, double, double) {
                    beta.rho() += data.getP(alpha.kind(), beta.kind())(r);
                });
        }

        cellList.clearGhosts(); // deletes ghosts, should not realloc
        cellList.makeGhosts();  // ghosts now have correct rho values
        cellList.updateHead();

        // std::cout << "out\n";

        for (std::size_t i = 0; i < numAtoms; ++i) {
            auto const &gamma = cellList[i];

            out[3 * i + 0] = 0;
            out[3 * i + 1] = 0;
            out[3 * i + 2] = 0;

            double const fpg = data.getF(gamma.kind()).grad(gamma.rho());

            // finds R^{\alpha\gamma}
            cellList.forEachNeigh(gamma, [&](auto const &alpha, double r,
                                             double dx, double dy, double dz) {
                double const fpa = data.getF(alpha.kind()).grad(alpha.rho());

                double mag = data.getV(alpha.kind(), gamma.kind()).grad(r);

                mag += fpg * data.getP(alpha.kind(), gamma.kind()).grad(r);

                mag += fpa * data.getP(gamma.kind(), alpha.kind()).grad(r);

                mag /= r;

                out[3 * i + 0] += dx * mag;
                out[3 * i + 1] += dy * mag;
                out[3 * i + 2] += dz * mag;
            });
        }
    }

    // // remaps into cell and performs sort
    // void sort(Eigen::ArrayXd &x) {
    //
    //     cellList.fillList(x);
    //
    //     std::sort(cellList.begin(), cellList.end(),
    //               [&](Atom const &a, Atom const &b) -> bool {
    //                   return box.lambda(a) < box.lambda(b);
    //               });
    //     // else
    //     // cj::sort_clever(list.begin(), list.end(), 0, box.numCells(),
    //     //                 [&](auto const &atom) { return box.lambda(atom);
    //     });
    //
    //     for (std::size_t i = 0; i < list.size(); ++i) {
    //         x[3 * i + 0] = list[i][0];
    //         x[3 * i + 1] = list[i][1];
    //         x[3 * i + 2] = list[i][2];
    //
    //         kinds[i] = list[i].kind();
    //     }
    // }

    template <typename T> auto quasiColourAll(T const &x) const {
        cellList.fillList(x);
        cellList.makeGhosts();
        cellList.updateHead();

        std::vector<std::size_t> colours;

        for (auto &&atom : cellList) {

            std::size_t count = atom.kind();

            cellList.forEachNeigh(
                atom, [&](auto const &neigh, double, double, double, double) {
                    count += NautyCanon::bonded(atom, neigh);
                });

            colours.push_back(count);
        }

        return colours;
    }
};
