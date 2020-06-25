#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Nauty.hpp"
#include "Rdf.hpp"
#include "utils.hpp"

struct TopoRef {
    std::vector<Eigen::Vector3d> ref{};

    std::size_t rdf_hash;
    std::array<std::size_t, 2> graph_hash;
    std::size_t count;

    bool operator==(TopoRef const &other) const {
        if (ref.size() != other.ref.size()) {
            return false;
        }
        if (rdf_hash != other.rdf_hash) {
            return false;
        }
        if (graph_hash != other.graph_hash) {
            return false;
        }

        for (std::size_t i = 0; i < ref.size(); ++i) {
            Eigen::Array3d dif = ref[i] - other.ref[i];
            if ((dif.abs() > 0.1).any()) {

                std::cout << i << ' ' << ref[i].transpose() << std::endl;
                std::cout << i << ' ' << other.ref[i].transpose() << std::endl;

                return false;
            }
        }

        return true;
    }

    TopoRef &operator+=(TopoRef const &other) {
        check(graph_hash == other.graph_hash, "cant add disimilar");
        check(rdf_hash == other.rdf_hash, "cant add disimilar");

        double total = count + other.count;

        for (std::size_t i = 0; i < ref.size(); ++i) {
            ref[i] = (ref[i] * count + other.ref[i] * other.count) / total;
        }

        count = total;

        return *this;
    }

    friend void to_json(nlohmann::json &j, TopoRef const &topo) {
        j = nlohmann::json{
            {"ref", topo.ref},
            {"rdf_hash", topo.rdf_hash},
            {"graph_hash", topo.graph_hash},
            {"count", topo.count},
        };
    }

    friend void from_json(nlohmann::json const &j, TopoRef &topo) {
        j.at("ref").get_to(topo.ref);
        j.at("rdf_hash").get_to(topo.rdf_hash);
        j.at("graph_hash").get_to(topo.graph_hash);
        j.at("count").get_to(topo.count);
    }
};

Eigen::Matrix3d modifiedGramSchmidtX(Eigen::Matrix3d const &in) {
    Eigen::Matrix3d out;

    out.col(0) = in.col(0);
    check(out.col(0).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(0).normalize();

    out.col(1) = in.col(1) - (in.col(1).adjoint() * out.col(0)) * out.col(0);
    check(out.col(1).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(1).normalize();

    out.col(2) = in.col(2) - (in.col(2).adjoint() * out.col(0)) * out.col(0);
    out.col(2) -= (out.col(2).adjoint() * out.col(1)) * out.col(1);
    check(out.col(2).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(2).normalize();

    using std::abs;

    check(abs(out.col(0).transpose() * out.col(1)) < 0.01, "gram schmidt fail");
    check(abs(out.col(1).transpose() * out.col(2)) < 0.01, "gram schmidt fail");
    check(abs(out.col(2).transpose() * out.col(0)) < 0.01, "gram schmidt fail");

    return out;
}

// Helper class holds Atom2 in LIST array of LCL with index of neighbours
class Atom2 {
  private:
    Eigen::Vector3d r;
    std::size_t idx;
    std::size_t n;

  public:
    Atom2(double x, double y, double z, std::size_t idx)
        : r{x, y, z}, idx{idx} {}
    Atom2(Atom2 const &) = default;
    Atom2(Atom2 &&) = default;
    Atom2() = default;
    Atom2 &operator=(Atom2 const &) = default;
    Atom2 &operator=(Atom2 &&) = default;

    inline std::size_t &next() { return n; }
    inline std::size_t next() const { return n; }

    inline std::size_t index() const { return idx; }

    inline Eigen::Vector3d const &pos() const { return r; }

    inline double &operator[](std::size_t i) { return r[i]; }
    inline double const &operator[](std::size_t i) const { return r[i]; }
};

class TopoClassify {
  private:
    Box box;

    std::size_t numAtom2s;

    Eigen::Array<std::size_t, Eigen::Dynamic, 1> head;
    std::vector<Atom2> list;

    std::vector<Rdf> rdfs;

    std::unordered_map<Rdf, TopoRef> catalog;

    static constexpr double BOND_DISTANCE = 2.66; // angstrom

  public:
    TopoClassify(std::size_t numAtom2s, Box const &box)
        : box{box}, numAtom2s{numAtom2s}, head(box.numCells()) {

        static_assert(std::is_trivially_destructible_v<Atom2>,
                      "can't clear Atom2s in constant time");

        static_assert(std::is_trivially_destructible_v<Rdf>,
                      "can't clear Atom2s in constant time");

        using nlohmann::json;
        json j = json::parse(std::ifstream("dump/toporef.json"));
        catalog = j.get<std::unordered_map<Rdf, TopoRef>>();
    }

  private:
    template <typename T> void fillCellList(T const &x3n) {
        check(static_cast<std::size_t>(x3n.size()) == 3 * numAtom2s,
              "wrong number of Atom2s");

        list.clear(); // should be order 1

        // add all Atom2s to cell list
        for (std::size_t i = 0; i < numAtom2s; ++i) {
            // map Atom2s into cell (0->xlen, 0->ylen, 0->zlen)
            auto [x, y, z] =
                box.mapIntoCell(x3n[3 * i + 0], x3n[3 * i + 1], x3n[3 * i + 2]);

            // check in cell
            check(x >= 0 && x < box.limits(0).len, "x out of box " << x);
            check(y >= 0 && y < box.limits(1).len, "y out of box " << y);
            check(z >= 0 && z < box.limits(2).len, "z out of box " << z);

            list.emplace_back(x, y, z, i / 3);
        }
    }

    // Alg 3.1
    void updateHead() {
        for (auto &&elem : head) {
            elem = list.size();
        }
        for (std::size_t i = 0; i < list.size(); ++i) {
            // update cell list
            Atom2 &atom = list[i];

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
                Atom2 atom = list[j];

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
    inline void findNeigh(Atom2 const &atom, F const &f) const {

        std::size_t const lambda = box.lambda(atom);
        std::size_t const end = list.size();
        double const cut_sq = box.rcut() * box.rcut();

        std::size_t index = head[lambda];

        // in same cell as Atom2
        do {
            Atom2 const &neigh = list[index];
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
        for (std::size_t off : box.getAdjOff()) {
            index = head[lambda + off];
            while (index != end) {
                // std::cout << "cell_t " << cell << std::endl;
                Atom2 const &neigh = list[index];
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
    template <typename T> void loadAtoms(T const &x) {
        fillCellList(x);
        makeGhosts();
        updateHead();

        rdfs.clear();

        check(numAtom2s * 3 == (std::size_t)x.size(), "wrong num atoms");

        for (auto atom = list.begin(); atom != list.begin() + numAtom2s;
             ++atom) {

            Rdf rdf{};

            findNeigh(*atom, [&](auto const &, double r, double, double,
                                 double) { rdf.add(r / box.rcut()); });

            rdfs.push_back(rdf);
        }
    }

    std::vector<Atom2> getNeighList(std::size_t centre) const {
        check(centre < numAtom2s, "out of bounds");

        std::vector<Atom2> ns;

        findNeigh(list[centre], [&](auto const &neigh, double, double, double,
                                    double) { ns.push_back(neigh); });

        return ns;
    }

    std::vector<Atom2> canonicalOrder(std::vector<Atom2> const &ns,
                                      std::array<std::size_t, 2> *hash) const {
        NautyGraph graph(ns.size());

        for (std::size_t i = 0; i < ns.size(); ++i) {
            for (std::size_t j = i + 1; j < ns.size(); ++j) {

                double sqdist = (ns[i].pos() - ns[j].pos()).squaredNorm();

                if (sqdist < BOND_DISTANCE * BOND_DISTANCE) {
                    graph.addEdge(i, j);
                }
            }
        }

        int const *order = graph.getCanonical();

        if (hash) {
            *hash = graph.hash();
        }

        std::vector<Atom2> ordered;

        std::transform(order, order + ns.size(), std::back_inserter(ordered),
                       [&](int o) { return ns[o]; });

        return ordered;
    }

    TopoRef classifyTopo(std::size_t idx) const {
        check(idx < numAtom2s, "out of bounds");

        TopoRef topo{
            {},
            std::hash<Rdf>{}(rdfs[idx]),
            {},
            1,
        };

        std::vector<Atom2> ns = getNeighList(idx);

        ns = canonicalOrder(ns, &topo.graph_hash);

        Eigen::Matrix3d transform = findBasis(ns);

        transform.transposeInPlace();

        Eigen::Vector3d origin = ns[0].pos();

        std::transform(ns.begin(), ns.end(), std::back_inserter(topo.ref),
                       [&](Atom2 const &atom) -> Eigen::Vector3d {
                           return transform * (atom.pos() - origin);
                       });

        return topo;
    }

    bool verify(Vector const &x) {
        for (std::size_t i = 0; i < rdfs.size(); ++i) {
            auto search = catalog.find(rdfs[i]);

            if (search == catalog.end()) {
                catalog.emplace(std::make_pair(rdfs[i], classifyTopo(i)));
            } else {
                TopoRef t = classifyTopo(i);

                if (!(search->second == t)) {
                    std::vector<std::size_t> col;

                    std::cout << search->second.graph_hash[0]
                              << search->second.graph_hash[1] << '\n';
                    std::cout << t.graph_hash[0] << t.graph_hash[1] << '\n';

                    // colour according to rdf hash
                    std::transform(rdfs.begin(), rdfs.end(),
                                   std::back_inserter(col), std::hash<Rdf>{});

                    // unprocessed topos colour 0
                    std::transform(col.begin() + i + 1, col.end(),
                                   col.begin() + i + 1,
                                   [=](std::size_t) { return 0; });

                    output(x, col);

                    auto bad = col[i];

                    // change color of topo collisions
                    std::transform(
                        col.begin(), col.begin() + i + 1, col.begin(),
                        [=](std::size_t h) { return h == bad ? 99 : h; });

                    col[i] = 100;

                    output(x, col);

                    check(false, "topology collison");

                    return false;
                } else {
                    search->second += t;
                }
            }
        }

        using nlohmann::json;

        json j = catalog;
        std::ofstream("dump/toporef.json") << j.dump(2);

        return true;
    }

    Eigen::Matrix3d findBasis(std::vector<Atom2> const &ns) const {

        Eigen::Vector3d origin = ns[0].pos();
        Eigen::Matrix3d basis;

        check(ns.size() > 2, "Not enough atoms to define basis");

        basis.col(0) = (ns[1].pos() - origin).normalized();

        // std::cout << "e0 @ 0->" << 1 << std::endl;

        std::size_t index_e1 = [&]() {
            for (std::size_t i = 2; i < ns.size(); ++i) {
                Eigen::Vector3d e1 = (ns[i].pos() - origin).normalized();

                Eigen::Vector3d cross = basis.col(0).cross(e1);

                // check for colinearity
                if (cross.squaredNorm() > 0.1) {
                    basis.col(1) = e1;
                    basis.col(2) = cross;
                    // std::cout << "e1 @ 0->" << i << std::endl;
                    return i;
                }
            }
            std::cerr << "All atoms colinear" << std::endl;
            std::terminate();
        }();

        for (std::size_t i = index_e1 + 1; i < ns.size(); ++i) {
            Eigen::Vector3d e2 = (ns[i].pos() - origin).normalized();

            double triple_prod = std::abs(basis.col(2).adjoint() * e2);

            // check for coplanarity with e0, e1
            if (triple_prod > 0.1) {
                basis.col(2) = e2;
                // std::cout << "e2 @ 0->" << i << std::endl;
                // std::cout << "Triple_prod " << triple_prod << std::endl;
                break;
            }
        }

        // std::cout << "\nTransformer (non-orthoganlaised) \n" << basis <<
        // std::endl;

        return modifiedGramSchmidtX(basis);
    }

    // Returns the index of the central atom for the mechanisim as well as the
    // reference data to reconstruct mechanisim.
    template <typename T>
    std::tuple<std::size_t, std::vector<Eigen::Vector3d>>
    classifyMech(Vector const &end) const {
        std::size_t centre = 0;

        { // find furthest moved
            double dr_sq_max = 0;

            for (std::size_t i = 0; i < numAtom2s; ++i) {

                Eigen::Vector3d end_pos{
                    end[3 * i + 0],
                    end[3 * i + 1],
                    end[3 * i + 2],
                };

                double dr_sq = (list[i].pos() - end_pos).squaredNorm();

                if (dr_sq > dr_sq_max) {
                    centre = i;
                    dr_sq_max = dr_sq;
                }
            }
        }

        std::vector<Atom2> near = getNeighList(centre);

        // std::vector<int> col(init.size() / 3, 0);
        // for (std::size_t i = 0; i < near.size(); ++i) {
        //     col[near[i].idx] = 1;
        // }
        // output(init, col);

        //////// find canonical-ordering ///////////

        near = canonicalOrder(near, nullptr);

        // for (std::size_t i = 0; i < near.size(); ++i) {
        //     col[near[i].idx] = i;
        // }
        //
        // output(init, col);

        ///////////// get basis vectors /////////////

        Eigen::Matrix3d transform = findBasis(near);

        transform.transposeInPlace();

        std::vector<Eigen::Vector3d> reference;

        std::transform(near.begin(), near.end(), std::back_inserter(reference),
                       [&](Atom2 const &atom) -> Eigen::Vector3d {
                           Eigen::Vector3d end_pos{
                               end[3 * atom.index() + 0],
                               end[3 * atom.index() + 1],
                               end[3 * atom.index() + 2],
                           };

                           return transform * (end_pos - atom.pos());
                       });

        // std::cout << "First atom delta: " << reference[0][0] << ' '
        //           << reference[0][1] << ' ' << reference[0][2] << "\n]\n";

        return {centre, std::move(reference)};
    }
};
