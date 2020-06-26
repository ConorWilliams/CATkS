#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include "Box.hpp"
#include "utils.hpp"

// Helper class to derive atoms from for use in cellList
class AtomBase {
  private:
    Eigen::Vector3d r;
    std::size_t n;

  public:
    AtomBase(double x, double y, double z) : r{x, y, z}, n{} {}

    inline std::size_t &next() { return n; }
    inline std::size_t next() const { return n; }

    inline Eigen::Vector3d const &pos() const { return r; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

template <typename Atom_t, typename Kind_t> class CellList {
  private:
    Box const &box;
    Kind_t const &kinds;

    Eigen::Array<std::size_t, Eigen::Dynamic, 1> head;
    std::vector<Atom_t> list;

  public:
    CellList(Box const &box, Kind_t const &kinds)
        : box{box}, kinds{kinds}, head(box.numCells()) {

        static_assert(std::is_trivially_destructible_v<Atom_t>,
                      "can't clear Atom_t in constant time");

        static_assert(std::is_base_of_v<AtomBase, Atom_t>,
                      "Atom_t must be derived from AtomBase");
    }

    inline std::size_t size() const { return kinds.size(); };

    inline auto begin() { return list.begin(); }
    inline auto begin() const { return list.begin(); }

    inline auto end() { return list.begin() + size(); }
    inline auto end() const { return list.begin() + size(); }

    inline Atom_t &operator[](std::size_t i) { return list[i]; }
    inline Atom_t const &operator[](std::size_t i) const { return list[i]; }

    template <typename T> void fillList(T const &x3n) {
        check(static_cast<std::size_t>(x3n.size()) == 3 * kinds.size(),
              "wrong number of atoms");

        list.clear(); // should be order 1

        // add all Atom2s to cell list
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            // map Atom2s into cell (0->xlen, 0->ylen, 0->zlen)
            auto [x, y, z] =
                box.mapIntoCell(x3n[3 * i + 0], x3n[3 * i + 1], x3n[3 * i + 2]);

            // check in cell
            check(x >= 0 && x < box.limits(0).len, "x out of box " << x);
            check(y >= 0 && y < box.limits(1).len, "y out of box " << y);
            check(z >= 0 && z < box.limits(2).len, "z out of box " << z);

            list.emplace_back(kinds[i], x, y, z, i);
        }
    }

    // Alg 3.1
    void updateHead() {
        for (auto &&elem : head) {
            elem = list.size();
        }
        for (std::size_t i = 0; i < list.size(); ++i) {
            // update cell list
            std::size_t lambda = box.lambda(list[i]);
            list[i].next() = head[lambda];
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
                Atom_t atom = list[j];

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
    inline void forEachNeigh(Atom_t const &atom, F &&f) const {

        std::size_t const lambda = box.lambda(atom);
        std::size_t const end = list.size();
        double const cut_sq = box.rcut() * box.rcut();

        std::size_t index = head[lambda];

        // in same cell as Atom2
        do {
            Atom_t const &neigh = list[index];

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
            // std::cout << "cell_t " << lambda + off << std::endl;
            while (index != end) {

                Atom_t const &neigh = list[index];

                double dx, dy, dz;

                double r_sq = box.normSq(neigh, atom, dx, dy, dz);

                if (r_sq <= cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }

                index = neigh.next();
            }
        }
    }
};
