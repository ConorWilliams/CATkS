#pragma once

#include <cmath>
#include <iomanip>
#include <type_traits>
#include <vector>

#include "Box.hpp"
#include "utils.hpp"

// Helper class to derive atoms from for use in cellList
class AtomSortBase {
  private:
    int k;
    Eigen::Vector3d r;
    std::size_t idx;

  public:
    AtomSortBase(int k, double x, double y, double z, std::size_t idx)
        : k{k}, r{x, y, z}, idx{idx} {}
    AtomSortBase() = default;

    inline std::size_t &index() { return idx; }
    inline std::size_t index() const { return idx; }

    inline Eigen::Vector3d const &pos() const { return r; }

    inline int const &kind() const { return k; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

template <typename Atom_t> class CellListSorted {
  private:
    struct Span {
        Atom_t *begin;
        Atom_t *end;
    };

    std::vector<int> const &kinds;
    std::vector<Span> head;

  protected:
    Box const &box;
    std::vector<Atom_t> list;

  public:
    CellListSorted(Box const &box, std::vector<int> const &kinds)
        : kinds{kinds}, head(box.numCells()), box{box} {

        static_assert(std::is_trivially_destructible_v<Atom_t>,
                      "can't clear Atom_t in constant time");

        static_assert(std::is_base_of_v<AtomSortBase, Atom_t>,
                      "Atom_t must be derived from AtomBase");
    }

    inline std::size_t size() const { return kinds.size(); };

    inline auto begin() { return list.begin(); }
    inline auto begin() const { return list.begin(); }

    inline auto end() { return list.begin() + size(); }
    inline auto end() const { return list.begin() + size(); }

    inline Atom_t &operator[](std::size_t i) { return list[i]; }
    inline Atom_t const &operator[](std::size_t i) const { return list[i]; }

  private:
    template <typename T> void insertAtoms(T const &x3n) {
        CHECK(static_cast<std::size_t>(x3n.size()) == 3 * kinds.size(),
              "wrong number of atoms");

        list.clear(); // should be order 1

        // add all Atom2s to cell list
        for (std::size_t i = 0; i < kinds.size(); ++i) {
            // map Atom2s into cell (0->xlen, 0->ylen, 0->zlen)
            auto [x, y, z] =
                box.mapIntoCell(x3n[3 * i + 0], x3n[3 * i + 1], x3n[3 * i + 2]);

            // CHECK in cell
            CHECK(x >= 0 && x < box.limits(0).len, "x out of box " << x);
            CHECK(y >= 0 && y < box.limits(1).len, "y out of box " << y);
            CHECK(z >= 0 && z < box.limits(2).len, "z out of box " << z);

            list.emplace_back(kinds[i], x, y, z, i);

            // CHECK(box.lambda(list.back()) == 13, "mapping fail");
        }
    }

    inline void sort() {
        std::sort(list.begin(), list.end(),
                  [&](Atom_t const &a, Atom_t const &b) -> bool {
                      return box.lambda(a) < box.lambda(b);
                  });
    }

    // Alg 3.1
    void updateHead() {
        std::fill(head.begin(), head.end(), Span{nullptr, nullptr});

        for (auto &&atom : list) {
            std::size_t l = box.lambda(atom);

            head[l].begin = head[l].begin == nullptr ? &atom : head[l].begin;
            head[l].end = (&atom) + 1;
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
                // TODO : atom->list[j] test
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

  public:
    template <typename T> void fill(T const &x3n) {
        insertAtoms(x3n);
        sort();
        makeGhosts();
        updateHead();
    }

    inline void rebuildGhosts() {
        list.resize(kinds.size());
        makeGhosts();
    }

    // applies f() for every neighbour (closer than rcut)
    template <typename F>
    inline void forEachNeigh(Atom_t const &atom, F &&f) const {

        std::size_t const l = box.lambda(atom);

        CHECK(l < box.numCells(), "bad lambda " << l);

        double const cut_sq = box.rcut() * box.rcut();

        std::for_each(head[l].begin, head[l].end, [&](Atom_t const &neigh) {
            if (&atom != &neigh) {
                double dx, dy, dz;
                double r_sq = box.normSq(neigh, atom, dx, dy, dz);
                if (r_sq < cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }
            }
        });

        // in adjecent cells -- don't need CHECK against self
        for (auto off : box.getAdjOff()) {

            std::size_t i = static_cast<long>(l) + off;

            std::for_each(head[i].begin, head[i].end, [&](Atom_t const &neigh) {
                double dx, dy, dz;
                double r_sq = box.normSq(neigh, atom, dx, dy, dz);
                if (r_sq < cut_sq) {
                    f(neigh, std::sqrt(r_sq), dx, dy, dz);
                }
            });
        }
    }
};
