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
#include "NautyFunc.hpp"
#include "utils.hpp"

inline constexpr double RCUT = 6; // angstrom

struct TopoRef {
    std::array<std::size_t, 2> hash;

    std::size_t count = 1;
    std::vector<Eigen::Vector3d> ref{};

    bool operator==(TopoRef const &other) const {
        if (ref.size() != other.ref.size()) {
            return false;
        }
        if (hash != other.hash) {
            return false;
        }

        for (std::size_t i = 0; i < ref.size(); ++i) {
            Eigen::Array3d dif = ref[i] - other.ref[i];
            // double tol = 0.7 / RCUT * (ref[i] - ref[0]).norm();

            // std::cout << (ref[i] - ref[0]).norm() << ' ' << tol << "\n";

            if ((dif.abs() > 0.1).any()) {

                std::cout << i << ' ' << ref[i].transpose() << std::endl;
                std::cout << i << ' ' << other.ref[i].transpose() << std::endl;

                return false;
            }
        }

        return true;
    }

    TopoRef &operator+=(TopoRef const &other) {
        check(hash == other.hash, "cant add disimilar");

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
            {"hash", topo.hash},
            {"count", topo.count},
        };
    }

    friend void from_json(nlohmann::json const &j, TopoRef &topo) {
        j.at("ref").get_to(topo.ref);
        j.at("hash").get_to(topo.hash);
        j.at("count").get_to(topo.count);
    }
};

Eigen::Matrix3d modifiedGramSchmidt(Eigen::Matrix3d const &in) {
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

template <typename T> Eigen::Matrix3d findBasis(std::vector<T> const &ns) {

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

    return modifiedGramSchmidt(basis);
}

template <typename Kind_t> class TopoClassify {
  private:
    class Atom : public AtomBase {
      private:
        std::size_t idx;

      public:
        Atom(int, double x, double y, double z, std::size_t idx)
            : AtomBase::AtomBase{x, y, z}, idx{idx} {}

        inline std::size_t index() const { return idx; }
    };

    CellList<Atom, Kind_t> cellList;

    std::vector<Graph> keys;
    std::vector<std::vector<Atom>> canon;

    Vector const *prev;

    std::unordered_map<Graph, TopoRef> catalog;

  public:
    TopoClassify(Box const &box, Kind_t const &kinds)
        : cellList{box, kinds}, keys{kinds.size()}, canon{kinds.size()} {

        static_assert(std::is_trivially_destructible_v<Graph>,
                      "can't clear graphs in constant time");

        constexpr auto fname = "dump/toporef.json";

        if (fileExist(fname)) {
            using nlohmann::json;
            json j = json::parse(std::ifstream());
            catalog = j.get<std::unordered_map<Graph, TopoRef>>();
        } else {
            std::cout << "Missing " << fname << std::endl;
        }
    }

    void colourPrint() const {
        output(*prev,
               transform_into(keys.begin(), keys.end(), std::hash<Graph>{}));
    }

    std::size_t size() const { return cellList.size(); };

    auto getRdf(std::size_t i) { return keys[i]; }
    auto getRdf(std::size_t i) const { return keys[i]; }

    void analyzeTopology(Vector &&x) = delete; // prevent tempories binding

    void analyzeTopology(Vector const &x) {
        // remember
        prev = &x;

        // Prime cell list
        cellList.fillList(x);
        cellList.makeGhosts();
        cellList.updateHead();

        std::vector<Atom> neigh;

        for (std::size_t i = 0; i < cellList.size(); ++i) {
            neigh.clear();
            // as forEach.. does not include centre
            neigh.push_back(cellList[i]);

            cellList.forEachNeigh(cellList[i],
                                  [&](auto const &n, double, double, double,
                                      double) { neigh.push_back(n); });

            canon[i].clear();

            keys[i] = canonicalize(neigh, canon[i]);
        }
    }

    TopoRef classifyTopo(std::size_t idx) const {
        check(idx < size(), "out of bounds");

        TopoRef topo{keys[idx].hash()};

        Eigen::Matrix3d transform = findBasis(canon[idx]);

        transform.transposeInPlace();

        Eigen::Vector3d origin = canon[idx][0].pos();

        transform_into(canon[idx].begin(), canon[idx].end(), topo.ref,
                       [&](Atom const &atom) -> Eigen::Vector3d {
                           return transform * (atom.pos() - origin);
                       });

        return topo;
    }

    bool verify() {
        for (std::size_t i = 0; i < size(); ++i) {
            auto search = catalog.find(keys[i]);

            // std::cout << i << ' ' << std::hash<Rdf>{}(rdfs[i]) <<
            // std::endl;

            if (search == catalog.end()) {
                catalog.emplace(std::make_pair(keys[i], classifyTopo(i)));
            } else {
                TopoRef t = classifyTopo(i);

                if (!(search->second == t)) {

                    // colour near;
                    std::vector<int> col2(prev->size() / 3, 0);

                    for (auto &&atom : canon[i]) {
                        col2[atom.index()] = 1;
                    }
                    output(*prev, col2);

                    std::vector<std::size_t> col;

                    // colour according to rdf hash
                    std::transform(keys.begin(), keys.end(),
                                   std::back_inserter(col), std::hash<Graph>{});

                    // unprocessed topos colour 0
                    std::transform(col.begin() + i + 1, col.end(),
                                   col.begin() + i + 1,
                                   [=](std::size_t) { return 0; });

                    output(*prev, col);

                    auto bad = col[i];

                    // change color of topo collisions
                    std::transform(
                        col.begin(), col.begin() + i + 1, col.begin(),
                        [=](std::size_t h) { return h == bad ? 99 : h; });

                    col[i] = 100;

                    output(*prev, col);

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

    // Returns the index of the central atom for the mechanisim as well as
    // the reference data to reconstruct mechanisim.
    std::tuple<std::size_t, std::vector<Eigen::Vector3d>>
    classifyMech(Vector const &end) const {
        std::size_t centre = 0;

        { // find furthest moved
            double dr_sq_max = 0;

            for (std::size_t i = 0; i < size(); ++i) {

                Eigen::Vector3d delta{
                    end[3 * i + 0] - (*prev)[3 * i + 0],
                    end[3 * i + 1] - (*prev)[3 * i + 1],
                    end[3 * i + 2] - (*prev)[3 * i + 2],
                };

                double dr_sq = delta.squaredNorm();

                if (dr_sq > dr_sq_max) {
                    centre = i;
                    dr_sq_max = dr_sq;
                }
            }

            // std::cout << "max move = " << std::sqrt(dr_sq_max) << '\n';
        }

        // std::vector<int> col(init.size() / 3, 0);
        //
        // for (std::size_t i = 0; i < near.size(); ++i) {
        //     col[near[i].index()] = 1;
        // }
        //
        // output(init, col);

        ////// find canonical-ordering ///////////

        //
        // for (std::size_t i = 0; i < near.size(); ++i) {
        //     col[near[i].index()] = i;
        // }
        //
        // col[near[0].index()] = 99;
        //
        // output(init, col);

        ///////////// get basis vectors /////////////

        Eigen::Matrix3d transform = findBasis(canon[centre]);

        transform.transposeInPlace();

        auto reference = transform_into(
            canon[centre].begin(), canon[centre].end(),
            std::vector<Eigen::Vector3d>{},
            [&](Atom const &atom) -> Eigen::Vector3d {
                Eigen::Vector3d delta{
                    end[3 * atom.index() + 0] - (*prev)[3 * atom.index() + 0],
                    end[3 * atom.index() + 1] - (*prev)[3 * atom.index() + 1],
                    end[3 * atom.index() + 2] - (*prev)[3 * atom.index() + 2],
                };

                return transform * delta;
            });

        return {centre, std::move(reference)};
    }

    Vector reconstruct(std::size_t idx,
                       std::vector<Eigen::Vector3d> const &ref) const {

        Vector end = *prev;

        check(canon[idx].size() == ref.size(),
              "wrong num atoms in reconstruction");

        // std::vector<int> col(init.size() / 3, 0);
        // for (std::size_t i = 0; i < near.size(); ++i) {
        //     check(near[i].idx < (int)col.size(), "bug in ordering alg
        //     again"); col[near[i].idx] = i + 1;
        // }
        //
        // output(init, col);

        Eigen::Matrix3d transform = findBasis(canon[idx]);

        for (std::size_t i = 0; i < canon[idx].size(); ++i) {
            Eigen::Vector3d delta = transform * ref[i];

            end[3 * canon[idx][i].index() + 0] += delta[0];
            end[3 * canon[idx][i].index() + 1] += delta[1];
            end[3 * canon[idx][i].index() + 2] += delta[2];
        }

        // output(end, col);

        return end;
    }
};
