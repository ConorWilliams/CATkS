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

#include "threadpool.hpp"
#include "utils.hpp"

inline constexpr double DELTA_E_TOL = 0.1; // TODO : make this 0.01 and test
inline constexpr double DIST_TOL = 0.2;

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

struct Mech {
    double active_E;
    double delta_E;
    std::size_t count;
    std::vector<Eigen::Vector3d> ref;

    Mech() = default; // required for json

    Mech(double active_E, double delta_E, std::vector<Eigen::Vector3d> &&ref)
        : active_E{active_E}, delta_E{delta_E}, count{1}, ref{std::move(ref)} {}

    friend void to_json(nlohmann::json &j, Mech const &mech) {
        j = nlohmann::json{
            {"active_E", mech.active_E},
            {"delta_E", mech.delta_E},
            {"count", mech.count},
            {"ref", mech.ref},
        };
    }

    friend void from_json(nlohmann::json const &j, Mech &mech) {
        j.at("active_E").get_to(mech.active_E);
        j.at("delta_E").get_to(mech.delta_E);
        j.at("count").get_to(mech.count);
        j.at("ref").get_to(mech.ref);
    }
};

class Topo {
    std::vector<Mech> mechs{}; // Mechs for this topology

  public:
    std::size_t count = 0;       // Num times occured in the simulation
    std::size_t sp_searches = 0; // Total sp searches initated from topology

    friend void to_json(nlohmann::json &j, Topo const &topo) {
        j = nlohmann::json{
            {"count", topo.count},
            {"sp_searches", topo.sp_searches},
            {"mechs", topo.mechs},
        };
    }

    friend void from_json(nlohmann::json const &j, Topo &topo) {
        j.at("count").get_to(topo.count);
        j.at("sp_searches").get_to(topo.sp_searches);
        j.at("mechs").get_to(topo.mechs);
    }

    inline std::vector<Mech> &getMechs() { return mechs; }

    bool pushMech(double active_E, double delta_E,
                  std::vector<Eigen::Vector3d> &&ref) {

        for (auto &&m : mechs) {
            CHECK(m.ref.size() == ref.size(), "topoolgy collison");

            if (std::abs(m.delta_E - delta_E) < DELTA_E_TOL &&
                std::abs(m.active_E - active_E) < DELTA_E_TOL) {

                std::size_t match_count = 0;

                for (std::size_t i = 0; i < ref.size(); ++i) {
                    Eigen::Array3d dif = ref[i] - m.ref[i];
                    if ((dif.abs() < DIST_TOL).all()) {
                        ++match_count;
                    }
                }

                if (match_count == ref.size()) {
                    // std::cout << "Exact match" << std::endl;

                    m.active_E =
                        (m.count * m.active_E + active_E) / (m.count + 1);

                    m.delta_E = (m.count * m.delta_E + delta_E) / (m.count + 1);

                    for (std::size_t i = 0; i < ref.size(); ++i) {
                        m.ref[i] =
                            (m.count * m.ref[i] + ref[i]) / (m.count + 1);
                    }
                    m.count += 1;

                    return false;
                }
            }
        }

        // std::cout << "New mech" << std::endl;

        mechs.emplace_back(active_E, delta_E, std::move(ref));

        return true;
    }
};

template <typename Canon> class Catalog {
  private:
    ////////////////////////////////////////////////////////////

    using Key_t = typename Canon::Key_t;

    ThreadPool pool;

    std::unordered_map<Key_t, Topo> catalog;

    static constexpr std::size_t sp_trys = 5;

    std::future<void> async_write;

  public:
    Catalog() : pool{std::thread::hardware_concurrency()} {
        using nlohmann::json;

        // std::string fname = "keys.lmc.json";
        //
        // if (fileExist(fname)) {
        //
        //     json keys = json::parse(std::ifstream(fname));
        //
        //     auto names = keys.get<std::unordered_map<Key_t, std::string>>();
        //
        //     for (auto &&[key, name] : names) {
        //         std::cout << "parsing: " << name << "\n";
        //         CHECK(fileExist(name), "missing a topo file");
        //
        //         json j = json::parse(std::ifstream(name));
        //
        //         catalog[key] = j.get<Topo>();
        //     }
        // } else {
        //     std::cout << "Missing " << fname << std::endl;
        // }

        std::string const fname = "catalog.json";

        if (fileExist(fname)) {
            std::cout << "Parsing: " << fname << std::endl;

            json j = json::parse(std::ifstream(fname));

            j.get_to(catalog);

        } else {
            std::cout << "Missing: " << fname << std::endl;
        }
    }

    inline auto &operator[](Key_t const &k) { return catalog[k].getMechs(); }

    inline auto const &operator[](Key_t const &k) const {
        return catalog[k].getMechs();
    }

    void write() {
        using nlohmann::json;

        if (async_write.valid()) {
            async_write.wait();
        }

        async_write = pool.execute([j = json(catalog)]() {
            ignore_result(std::system("mv catalog.json catalog.json.bak"));
            std::ofstream("catalog.json") << j << std::endl;
        });
    }

    template <typename F, typename C, typename MinImage>
    int update(Vector const &x, F const &f, C const &cl, MinImage const &mi,
               std::vector<std::size_t> const &idxs) {

        using result_t = std::vector<std::tuple<Vector, Vector>>;

        std::cout << "Launching " << idxs.size() * sp_trys << " sp searches.\n";

        double f_x = f(x);

        std::size_t sps = 0;
        std::size_t new_mechs = 0;

        std::vector<std::future<result_t>> futures;

        for (auto &&idx : idxs) {
            futures.emplace_back(pool.execute(
                [=, &x, &mi]() { return findSaddle(sp_trys, x, idx, f, mi); }));
        }

        for (auto &&vec : futures) {
            for (auto &&[sp, end] : vec.get()) {

                ++sps;

                auto [centre, ref] = cl.classifyMech(end);

                double barrier = f(sp) - f_x;
                double delta = f(end) - f_x;

                CHECK(barrier > 0, "found a negative energy sp?");

                if (catalog[cl[centre]].pushMech(barrier, delta,
                                                 std::move(ref))) {
                    new_mechs += 1;
                }
            }
        }
        //}

        if (idxs.size() > 0) {
            std::cout << sps << '/' << idxs.size() * sp_trys
                      << " searches found saddles.\n";

            std::cout << new_mechs << '/' << sps
                      << " saddles identified new mechanisims.\n";
        }

        return new_mechs;
    }

    template <typename C> std::vector<std::size_t> getSearchIdxs(C const &cl) {

        std::vector<std::size_t> idxs;

        for (std::size_t i = 0; i < cl.size(); ++i) {

            catalog[cl[i]].count += 1; // default constructs new

            while (catalog[cl[i]].sp_searches < 25 ||
                   ipow<3>(catalog[cl[i]].sp_searches) < catalog[cl[i]].count) {

                catalog[cl[i]].sp_searches += sp_trys;

                idxs.push_back(i);
            }
        }

        return idxs;
    }
};
