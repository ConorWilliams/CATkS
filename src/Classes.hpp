#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

#include "Rdf.hpp"
#include "utils.hpp"

inline constexpr double DELTA_E_TOL = 0.01;
inline constexpr double DIST_TOL = 0.1;

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
    double active_E{};
    double delta_E{};
    std::size_t count{};
    std::vector<Eigen::Vector3d> ref{};

    Mech() = default;

    Mech(double active_E, double delta_E, std::vector<Eigen::Vector3d> &&ref)
        : active_E{active_E}, delta_E{delta_E}, count{1}, ref{std::move(ref)} {}

    friend void to_json(nlohmann::json &j, Mech const &mech) {
        j = nlohmann::json{{"active_E", mech.active_E},
                           {"delta_E", mech.delta_E},
                           {"count", mech.count},
                           {"ref", mech.ref}};
    }

    friend void from_json(nlohmann::json const &j, Mech &mech) {
        j.at("active_E").get_to(mech.active_E);
        j.at("delta_E").get_to(mech.delta_E);
        j.at("count").get_to(mech.count);
        j.at("ref").get_to(mech.ref);
    }
};

class Topology {

    std::vector<Mech> mechs{}; // Mechs for this topology

  public:
    std::size_t count = 0;       // Num times occured in the simulation
    std::size_t sp_searches = 0; // Total sp searches initated from topology
    std::size_t count_mechs = 0;

    friend void to_json(nlohmann::json &j, Topology const &topo) {
        j = nlohmann::json{
            {"count", topo.count},
            {"sp_searches", topo.sp_searches},
            {"count_mechs", topo.count_mechs},
            {"mechs", topo.mechs},
        };
    }

    friend void from_json(nlohmann::json const &j, Topology &topo) {
        j.at("count").get_to(topo.count);
        j.at("sp_searches").get_to(topo.sp_searches);
        j.at("count_mechs").get_to(topo.count_mechs);
        j.at("mechs").get_to(topo.mechs);
    }

    inline std::vector<Mech> const &getMechs() const { return mechs; }

    void pushMech(double active_E, double delta_E,
                  std::vector<Eigen::Vector3d> &&ref) {

        for (auto &&m : mechs) {
            check(m.ref.size() == ref.size(), "topoolgy collison");

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
                    std::cout << "Exact match" << std::endl;

                    m.active_E =
                        (m.count * m.active_E + active_E) / (m.count + 1);

                    m.delta_E = (m.count * m.delta_E + delta_E) / (m.count + 1);

                    for (std::size_t i = 0; i < ref.size(); ++i) {
                        m.ref[i] =
                            (m.count * m.ref[i] + ref[i]) / (m.count + 1);
                    }
                    m.count += 1;
                    return;
                }
            }
        }

        std::cout << "New mech" << std::endl;

        ++count_mechs;

        mechs.emplace_back(active_E, delta_E, std::move(ref));
    }
};

std::unordered_map<Rdf, Topology> readMap(std::string const &dir) {
    using nlohmann::json;

    std::unordered_map<Rdf, Topology> map;

    json rdfs = json::parse(std::ifstream(dir + "/rdf.json"));

    for (auto &&elem : rdfs) {

        Rdf r = elem.get<Rdf>();

        // std::cout << dir + '/' + r.to_string() << std::endl;

        json j = json::parse(std::ifstream(dir + '/' + r.to_string()));

        map[r] = j.get<Topology>();
    }

    return map;
}

void writeMap(std::string const &dir,
              std::unordered_map<Rdf, Topology> const &map) {

    using nlohmann::json;

    json keys;

    for (auto &&[rdf, topo] : map) {
        json j = topo;
        std::ofstream(dir + '/' + rdf.to_string()) << j.dump(2);
        keys.push_back(rdf);
    }

    std::ofstream(dir + "/rdf.json") << keys.dump(2);
}
