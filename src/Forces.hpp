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
#include "utils.hpp"

#include "Canon.hpp"
#include "Cell.hpp"

class FuncEAM2 {
  private:
    // Helper class holds atom in LIST array of LCL with index of neighbours
    class Atom : public AtomSortBase {
      private:
        double m_fP;

      public:
        using AtomSortBase::AtomSortBase;
        inline double &fP() { return m_fP; }
        inline double fP() const { return m_fP; }
    };

    mutable CellListSorted<Atom> cellList;

    TabEAM const &data;

    std::size_t numAtoms;

  public:
    FuncEAM2(Box const &box, std::vector<int> const &kinds, TabEAM const &data)
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

        cellList.fill(x); // initilises rho to zeros

        double sum = 0;

        for (auto &&alpha : cellList) {

            double rho = 0;

            cellList.forEachNeigh(
                alpha, [&](auto const &beta, double r, double, double, double) {
                    sum += 0.5 * data.getV(alpha.kind(), beta.kind())(r);

                    rho += data.getP(beta.kind(), alpha.kind())(r);
                });

            sum += data.getF(alpha.kind())(rho);
        }
        return sum;
    }

    // computes grad
    template <typename T> void operator()(T const &x, Vector &out) const {

        cellList.fill(x);

        for (auto &&beta : cellList) {
            double rho = 0;
            // Computes rho at atom
            cellList.forEachNeigh(
                beta, [&](auto const &alpha, double r, double, double, double) {
                    rho += data.getP(alpha.kind(), beta.kind())(r);
                });
            // Compute F'(rho) at atom
            beta.fP() = data.getF(beta.kind()).grad(rho);
        }

        cellList.rebuildGhosts();

        for (std::size_t i = 0; i < numAtoms; ++i) {
            auto const &gamma = cellList[i];

            out[3 * gamma.index() + 0] = 0;
            out[3 * gamma.index() + 1] = 0;
            out[3 * gamma.index() + 2] = 0;

            // finds R^{\alpha\gamma}
            cellList.forEachNeigh(gamma, [&](auto const &alpha, double r,
                                             double dx, double dy, double dz) {
                double mag = data.getV(alpha.kind(), gamma.kind()).grad(r);

                mag +=
                    gamma.fP() * data.getP(alpha.kind(), gamma.kind()).grad(r);

                mag +=
                    alpha.fP() * data.getP(gamma.kind(), alpha.kind()).grad(r);

                mag *= 1 / r;

                out[3 * gamma.index() + 0] += dx * mag;
                out[3 * gamma.index() + 1] += dy * mag;
                out[3 * gamma.index() + 2] += dz * mag;
            });
        }
    }

    template <typename T> auto quasiColourAll(T const &x) const {
        cellList.fill(x);

        std::vector<int> colours(x.size() / 3);

        for (auto &&atom : cellList) {

            int count = atom.kind();

            if (atom.kind() == 1) {
                colours[atom.index()] = 99;

            } else {
                cellList.forEachNeigh(
                    atom, [&](auto const &neigh, double, double, double,
                              double) { count += bonded(atom, neigh); });

                colours[atom.index()] = count;
            }
        }

        return colours;
    }
};
