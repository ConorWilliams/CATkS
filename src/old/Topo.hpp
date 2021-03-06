#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "nlohmann/json.hpp"

#include "Box.hpp"
#include "Catalog.hpp"
#include "Cell.hpp"
#include "DumpXYX.hpp"
#include "utils.hpp"

// inline constexpr double MOVE_TOL = 0.1; // angstrom

Eigen::Matrix3d modifiedGramSchmidt(Eigen::Matrix3d const &in) {
    Eigen::Matrix3d out;

    out.col(0) = in.col(0);
    CHECK(out.col(0).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(0).normalize();

    out.col(1) = in.col(1) - (in.col(1).adjoint() * out.col(0)) * out.col(0);
    CHECK(out.col(1).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(1).normalize();

    out.col(2) = in.col(2) - (in.col(2).adjoint() * out.col(0)) * out.col(0);
    out.col(2) -= (out.col(2).adjoint() * out.col(1)) * out.col(1);
    CHECK(out.col(2).squaredNorm() > 0.01, "linear dependance in basis");
    out.col(2).normalize();

    using std::abs;

    CHECK(abs(out.col(0).transpose() * out.col(1)) < 0.01, "gram schmidt fail");
    CHECK(abs(out.col(1).transpose() * out.col(2)) < 0.01, "gram schmidt fail");
    CHECK(abs(out.col(2).transpose() * out.col(0)) < 0.01, "gram schmidt fail");

    return out;
}

template <typename T> Eigen::Matrix3d findBasis(std::vector<T> const &ns) {

    Eigen::Vector3d origin = ns[0].pos();
    Eigen::Matrix3d basis;

    CHECK(ns.size() > 3, "Not enough atoms to define basis");

    basis.col(0) = (ns[1].pos() - origin).normalized();

    // std::cout << "e0 @ 0->" << 1 << std::endl;

    std::size_t index_e1 = [&]() {
        for (std::size_t i = 2; i < ns.size(); ++i) {
            Eigen::Vector3d e1 = (ns[i].pos() - origin).normalized();

            Eigen::Vector3d cross = basis.col(0).cross(e1);

            // CHECK for colinearity
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

        // CHECK for coplanarity with e0, e1
        if (triple_prod > 0.1) {
            basis.col(2) = e2;
            // std::cout << "e2 @ 0->" << i << std::endl;
            // std::cout << "Triple_prod " << triple_prod << std::endl;
            break;
        }
    }

    return modifiedGramSchmidt(basis);
}
namespace detail {

struct Topo {
    std::array<std::size_t, 2> hash;

    std::size_t count = 1;
    std::vector<Eigen::Vector3d> ref{};

    bool operator==(Topo const &other) const {
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

            if ((dif.abs() > DIST_TOL).any()) {

                std::cout << i << ' ' << ref[i].transpose() << std::endl;
                std::cout << i << ' ' << other.ref[i].transpose() << std::endl;

                return false;
            }
        }

        return true;
    }

    Topo &operator+=(Topo const &other) {
        CHECK(hash == other.hash, "cant add disimilar");

        double total = count + other.count;

        for (std::size_t i = 0; i < ref.size(); ++i) {
            ref[i] = (ref[i] * count + other.ref[i] * other.count) / total;
        }

        count = total;

        return *this;
    }

    friend void to_json(nlohmann::json &j, Topo const &topo) {
        j = nlohmann::json{
            {"ref", topo.ref},
            {"hash", topo.hash},
            {"count", topo.count},
        };
    }

    friend void from_json(nlohmann::json const &j, Topo &topo) {
        j.at("ref").get_to(topo.ref);
        j.at("hash").get_to(topo.hash);
        j.at("count").get_to(topo.count);
    }
};

} // namespace detail

class Atom : public AtomBase {
  private:
    std::size_t idx;

  public:
    Atom(int k, double x, double y, double z, std::size_t idx)
        : AtomBase::AtomBase{k, x, y, z}, idx{idx} {}

    inline std::size_t index() const { return idx; }
};

template <typename Canon> class TopoClassify : protected CellList<Atom> {
  private:
    using Key_t = typename Canon::Key_t;

    std::vector<Key_t> keys;
    std::vector<std::vector<Atom>> canon;

    Vector const *prev;

    std::unordered_map<Key_t, detail::Topo> catalog;

    static constexpr auto fname = "toporef.json";

    detail::Topo classifyTopo(std::size_t idx) const {
        CHECK(idx < size(), "out of bounds");

        detail::Topo topo{keys[idx].hash()};

        Eigen::Matrix3d transform = findBasis(canon[idx]);

        transform.transposeInPlace();

        Eigen::Vector3d origin = canon[idx][0].pos();

        transform_into(canon[idx].begin(), canon[idx].end(), topo.ref,
                       [&](Atom const &atom) -> Eigen::Vector3d {
                           return transform * (atom.pos() - origin);
                       });

        return topo;
    }

  public:
    TopoClassify(Box const &box, std::vector<int> const &kinds)
        : CellList::CellList{box, kinds}, keys{kinds.size()},
          canon{kinds.size()} {

        static_assert(std::is_trivially_destructible_v<Key_t>,
                      "can't clear graphs in constant time");

        if (fileExist(fname)) {
            std::cout << "Parsing: " << fname << std::endl;
            using nlohmann::json;
            json j = json::parse(std::ifstream(fname));
            catalog = j.get<std::unordered_map<Key_t, detail::Topo>>();
        } else {
            std::cout << "Missing: " << fname << std::endl;
        }
    }

    using CellList::size;

    void colourPrint() const {
        output(*prev,
               transform_into(keys.begin(), keys.end(), std::hash<Key_t>{}));
    }

    inline Key_t &operator[](std::size_t i) { return keys[i]; }
    inline Key_t const &operator[](std::size_t i) const { return keys[i]; }

    void analyzeTopology(Vector &&x) = delete; // prevent tempories binding

    void analyzeTopology(Vector const &x) {
        // remember
        prev = &x;

        // Prime cell list
        fillList(x);
        makeGhosts();
        updateHead();

        std::vector<Atom> neigh;

        for (std::size_t i = 0; i < size(); ++i) {
            neigh.clear();
            // as forEach.. does not include centre
            neigh.push_back(list[i]);

            forEachNeigh(list[i], [&](auto const &n, double, double, double,
                                      double) { neigh.push_back(n); });

            canon[i].clear();

            keys[i] = Canon::canonicalize(neigh, canon[i]);
        }
    }

    int verify() {
        std::size_t count_new = 0;

        for (std::size_t i = 0; i < size(); ++i) {
            auto search = catalog.find(keys[i]);

            // std::cout << i << ' ' << std::hash<Rdf>{}(rdfs[i]) <<
            // std::endl;

            detail::Topo t = classifyTopo(i);

            if (search == catalog.end()) {
                ++count_new;
                catalog.emplace(std::make_pair(keys[i], std::move(t)));

            } else if (!(search->second == t)) {
                // colour near;
                std::vector<int> col2(prev->size() / 3, 0);

                for (auto &&atom : canon[i]) {
                    col2[atom.index()] = 1;
                }
                output(*prev, col2);

                for (std::size_t j = 0; j < canon[i].size(); ++j) {
                    col2[canon[i][j].index()] = j;
                }

                col2[canon[i][0].index()] = 99;

                // colour by label order
                output(*prev, col2);

                // colour according to rdf hash
                auto col = transform_into(keys.begin(), keys.end(),
                                          std::hash<Key_t>{});

                output(*prev, col);

                // unprocessed topos colour 0
                std::transform(col.begin() + i + 1, col.end(),
                               col.begin() + i + 1,
                               [=](std::size_t) { return 0; });

                output(*prev, col);

                auto bad = col[i];

                std::vector<std::size_t> other;
                // change col of collisons
                for (std::size_t j = 0; j < col.size(); ++j) {
                    if (col[j] == bad && i != j) {
                        col[j] = 99;
                        other.push_back(j);
                    }
                }

                col[i] = 99;

                output(*prev, col);

                for (std::size_t idx : other) {
                    // colour near;
                    std::vector<int> col(prev->size() / 3, 0);

                    for (auto &&atom : canon[idx]) {
                        col[atom.index()] = 1;
                    }
                    output(*prev, col);
                }

                VERIFY(false, "topology collison");

                return -1;
            } else {
                search->second += t;
            }
        }

        std::cout << "All topos verified! ";

        std::cout << count_new << '/' << size() << " are new.\n";

        return count_new;
    }

    void write() {
        using nlohmann::json;

        json j = catalog;
        std::ofstream(fname) << j.dump(2);
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
        }
        // { // verify all active atoms are within r_topo of centre

        Eigen::Matrix3d transform = findBasis(canon[centre]);

        transform.transposeInPlace();

        return {centre,
                transform_into(canon[centre].begin(), canon[centre].end(),
                               [&](Atom const &atom) -> Eigen::Vector3d {
                                   auto i = 3 * atom.index();

                                   Eigen::Vector3d delta{
                                       end[i + 0] - (*prev)[i + 0],
                                       end[i + 1] - (*prev)[i + 1],
                                       end[i + 2] - (*prev)[i + 2],
                                   };

                                   return transform * delta;
                               })};
    }

    Vector reconstruct(std::size_t idx,
                       std::vector<Eigen::Vector3d> const &ref) const {

        Vector end = *prev;

        CHECK(canon[idx].size() == ref.size(),
              "wrong num atoms in reconstruction");

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
