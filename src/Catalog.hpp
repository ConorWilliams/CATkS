
#pragma once

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

#include "Cell.hpp"
#include "threadpool.hpp"
#include "utils.hpp"

inline constexpr double DELTA_E_TOL = 0.1; // TODO : make this 0.01 and test
inline constexpr double DIST_TOL = 0.2;

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

namespace nlohmann {
template <> struct adl_serializer<Eigen::Vector3d> {
    static void to_json(json &j, Eigen::Vector3d const &vec) {
        j = nlohmann::json{vec[0], vec[1], vec[2]};
    }

    static void from_json(json const &j, Eigen::Vector3d &vec) {
        vec[0] = j.at(0);
        vec[1] = j.at(1);
        vec[2] = j.at(2);
    }
};
} // namespace nlohmann

struct VecCoord {

    std::vector<Eigen::Vector3d> data{};
    std::size_t count = 1;

    friend void to_json(nlohmann::json &j, VecCoord const &other) {
        j = nlohmann::json{
            {"data", other.data},
            {"count", other.count},
        };
    }

    friend void from_json(nlohmann::json const &j, VecCoord &other) {
        j.at("data").get_to(other.data);
        j.at("count").get_to(other.count);
    }

    bool operator==(VecCoord const &other) const {
        if (data.size() != other.data.size()) {
            return false;
        }

        for (std::size_t i = 0; i < data.size(); ++i) {
            Eigen::Array3d dif = data[i] - other.data[i];

            if ((dif.abs() > DIST_TOL).any()) {
                return false;
            }
        }

        return true;
    }

    VecCoord &operator+=(VecCoord const &other) {

        double total = count + other.count;

        for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] = (data[i] * count + other.data[i] * other.count) / total;
        }

        count = total;

        return *this;
    }
};

struct Mechanism {
    double active_E;
    double delta_E;
    VecCoord ref;

    Mechanism() = default; // required for json

    Mechanism(double active_E, double delta_E,
              std::vector<Eigen::Vector3d> &&ref)
        : active_E{active_E}, delta_E{delta_E}, ref{std::move(ref)} {}

    friend void to_json(nlohmann::json &j, Mechanism const &mech) {
        j = nlohmann::json{
            {"active_E", mech.active_E},
            {"delta_E", mech.delta_E},
            {"ref", mech.ref},
        };
    }

    friend void from_json(nlohmann::json const &j, Mechanism &mech) {
        j.at("active_E").get_to(mech.active_E);
        j.at("delta_E").get_to(mech.delta_E);
        j.at("ref").get_to(mech.ref);
    }
};

struct Topology {
    VecCoord ref{};                 // Reference data for verification
    std::vector<Mechanism> mechs{}; // Mechanisms for this topology
    std::size_t sp_searches = 0;    // Total sp searches initated from topology

    friend void to_json(nlohmann::json &j, Topology const &topo) {
        j = nlohmann::json{
            {"mechs", topo.mechs},
            {"ref", topo.ref},
            {"sp_searches", topo.sp_searches},

        };
    }

    friend void from_json(nlohmann::json const &j, Topology &topo) {
        j.at("mechs").get_to(topo.mechs);
        j.at("ref").get_to(topo.ref);
        j.at("sp_searches").get_to(topo.sp_searches);
    }

    bool pushMech(Mechanism &&mech) {
        for (auto &&m : mechs) {
            if (m.ref == mech.ref) {
                m.ref += mech.ref;
                return 0;
            }
        }
        mechs.push_back(std::move(mech));
        return true;
    }
};

template <typename Canon> class Catalog {
  private:
    using Key_t = typename Canon::Key_t;

    struct Atom : public AtomSortBase {
        using AtomSortBase::AtomSortBase;
    };

    static_assert(std::is_trivially_destructible_v<Key_t>, "");

    static constexpr std::size_t SPS_PER_THREAD = 5;
    static constexpr std::size_t MIN_SPS = 4 * SPS_PER_THREAD;
    static constexpr char FNAME[] = "catalog.json";

    ThreadPool pool;
    CellListSorted<Atom> cell_list;

    std::unordered_map<Key_t, Topology> catalog;

    std::vector<Key_t> keys;
    std::vector<std::vector<std::size_t>> orders;
    std::vector<Eigen::Matrix3d> transforms;

    std::future<void> async_write;

  public:
    Catalog(Box const &box, std::vector<int> const &kinds)
        : pool{std::thread::hardware_concurrency()}, cell_list{box, kinds},
          keys{kinds.size()}, orders{kinds.size()}, transforms{kinds.size()} {

        if (fileExist(FNAME)) {
            std::cout << "Parsing: " << FNAME << std::endl;

            nlohmann::json j = nlohmann::json::parse(std::ifstream(FNAME));

            j.get_to(catalog);

        } else {
            std::cout << "Missing: " << FNAME << std::endl;
        }
    }

    Topology &operator[](std::size_t i) { return catalog[keys[i]]; }

    Topology const &operator[](std::size_t i) const {
        return catalog.at(keys[i]);
    }

    inline std::size_t size() const { return cell_list.size(); }

    void write() {
        // Must wait for prev write to complete
        if (async_write.valid()) {
            async_write.wait();
        }

        // Garentee back up incase of exit()
        ignore_result(std::system("mv catalog.json catalog.json.bak"));

        async_write = pool.execute([j = nlohmann::json(catalog)]() {
            std::ofstream("catalog.json") << j << std::endl;
        });
    }

    // Launches SP searches for every new topology in x, processes the resulting
    // mechanisms and merges them into the catalog.
    template <typename F, typename MinImage>
    int update(Vector const &x, F const &f, MinImage const &mi) {

        auto [freq, count] = analyzeTopology(x);

        std::cout << "Collision frequency @ " << freq << "%\n";
        std::cout << count << '/' << size() << " new topos, launched ";

        // Result of findSaddle(...)
        using result_t = std::vector<std::tuple<Vector, Vector>>;
        //
        std::vector<std::future<result_t>> futures;

        // Launch SP searches asynchronously

        for (std::size_t i = 0; i < size(); ++i) {
            Topology &topo = catalog[keys[i]];

            while (topo.sp_searches < MIN_SPS ||
                   ipow<3>(topo.sp_searches) < topo.ref.count) {

                topo.sp_searches += SPS_PER_THREAD;

                futures.emplace_back(pool.execute([=, &x, &mi]() {
                    return findSaddle(SPS_PER_THREAD, x, i, f, mi);
                }));
            }
        }

        std::cout << futures.size() << " threads\n";

        std::size_t launched = futures.size() * SPS_PER_THREAD;
        std::size_t sucessful = 0;
        std::size_t new_mechs = 0;

        double f_x = f(x);

        // Process result of SP searches

        for (auto &&vec : futures) {
            for (auto &&[sp, end] : vec.get()) {

                ++sucessful;

                double barrier = f(sp) - f_x;
                double delta = f(end) - f_x;

                auto [centre, mech] = makeMech(barrier, delta, x, end);

                CHECK(barrier > 0, "found a negative energy sp?");

                new_mechs += catalog[keys[centre]].pushMech(std::move(mech));
            }
        }

        if (launched > 0) {

            std::cout << sucessful << '/' << launched
                      << " searches found conected SPs.\n";

            std::cout << new_mechs << '/' << sucessful
                      << " SPS discovered new mechanisms.\n";
        }

        return new_mechs;
    }

    // Reconstruct mechanism at atom idx onto x
    void reconstruct(std::size_t idx, Vector &x, Mechanism const &mech) const {

        CHECK(orders[idx].size() == mech.ref.data.size(),
              "wrong num atoms in reconstruction " << orders[idx].size() << ' '
                                                   << mech.ref.data.size());

        Eigen::Matrix3d Tr = transforms[idx].transpose();

        for (std::size_t i = 0; i < orders[idx].size(); ++i) {
            Eigen::Vector3d delta = Tr * mech.ref.data[i];

            x[3 * orders[idx][i] + 0] += delta[0];
            x[3 * orders[idx][i] + 1] += delta[1];
            x[3 * orders[idx][i] + 2] += delta[2];
        }
    }

    // For each atom processes its topology such that catalog contains all
    // topologies in x. Also stores the key, transform-matrix & canonicle
    // neighbour order into the corresponding vectors.
    std::pair<double, std::size_t> analyzeTopology(Vector const &x) {
        cell_list.fill(x);

        static std::vector<Atom> neigh;
        static std::vector<Atom> canon;

        std::size_t new_topo = 0;
        std::size_t reclassify = 0;

        for (auto &&atom : cell_list) {

            neigh.clear();
            // as forEachNeigh(atom, ...) does not include atom
            neigh.push_back(atom);

            cell_list.forEachNeigh(
                atom, [&](auto const &n, auto &&...) { neigh.push_back(n); });

            std::size_t idx = atom.index();

            auto &&[novel, lvl] = merge(idx, neigh, canon);

            new_topo += novel;
            reclassify += lvl;

            orders[idx].clear();
            for (auto &&atom : canon) {
                orders[idx].push_back(atom.index());
            }
        }

        return {100.0 * reclassify / size(), new_topo};
    }

  private:
    // Refines topolgial classification until unique position in catalog.
    // return.first true if new topology, return.second is classifcation level
    // required.
    std::pair<bool, std::size_t>
    merge(std::size_t idx, std::vector<Atom> &neigh, std::vector<Atom> &canon) {

        static Topology topo;

        for (std::size_t lvl = 0;; ++lvl) {
            canon.clear();

            keys[idx] = Canon::canonicalize(neigh, canon, lvl);

            transforms[idx] = findBasis(canon).transpose();

            topo.ref.data.clear();
            for (auto &&atom : canon) {
                topo.ref.data.emplace_back(transforms[idx] *
                                           (atom.pos() - canon[0].pos()));
            }

            auto &&[it, inserted] = catalog.insert({keys[idx], topo});

            if (inserted) {
                // New topology
                return {true, lvl};
            } else if (it->second.ref == topo.ref) {
                // Existing topology
                it->second.ref += topo.ref;
                return {false, lvl};
            } else if (lvl > neigh.size()) {
                // topo collision
                std::cout << "topo overflow @ lvl: " << lvl << std::endl;
                std::terminate();
            }
        }
    }

    // Returns the index of the central atom for the mechanisim and the
    // reference data to reconstruct mechanisim.
    std::pair<std::size_t, Mechanism> makeMech(double barrier, double delta,
                                               Vector const &beg,
                                               Vector const &end) const {
        std::size_t centre = 0;

        { // find furthest moved
            double dr_sq_max = 0;

            for (std::size_t i = 0; i < size(); ++i) {

                Eigen::Vector3d delta{
                    end[3 * i + 0] - beg[3 * i + 0],
                    end[3 * i + 1] - beg[3 * i + 1],
                    end[3 * i + 2] - beg[3 * i + 2],
                };

                double dr_sq = delta.squaredNorm();

                if (dr_sq > dr_sq_max) {
                    centre = i;
                    dr_sq_max = dr_sq;
                }
            }
        }

        // TODO: check all moved within r_topo of centre.

        return {
            centre,
            {
                barrier,
                delta,
                transform_into(orders[centre].begin(), orders[centre].end(),
                               [&](std::size_t i) -> Eigen::Vector3d {
                                   return transforms[centre] *
                                          Eigen::Vector3d{
                                              end[3 * i + 0] - beg[3 * i + 0],
                                              end[3 * i + 1] - beg[3 * i + 1],
                                              end[3 * i + 2] - beg[3 * i + 2],
                                          };
                               }),
            },
        };
    }
};
