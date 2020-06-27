#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

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

template <typename Canon> class Catalog {
  private:
    class Topo {
        struct Mech {
            double active_E{};
            double delta_E{};
            std::size_t count{};
            std::vector<Eigen::Vector3d> ref{};

            Mech() = default;

            Mech(double active_E, double delta_E,
                 std::vector<Eigen::Vector3d> &&ref)
                : active_E{active_E}, delta_E{delta_E}, count{1}, ref{std::move(
                                                                      ref)} {}

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

        std::vector<Mech> mechs{}; // Mechs for this topology

      public:
        std::size_t count = 0;       // Num times occured in the simulation
        std::size_t sp_searches = 0; // Total sp searches initated from topology
        std::size_t count_mechs = 0;

        friend void to_json(nlohmann::json &j, Topo const &topo) {
            j = nlohmann::json{
                {"count", topo.count},
                {"sp_searches", topo.sp_searches},
                {"count_mechs", topo.count_mechs},
                {"mechs", topo.mechs},
            };
        }

        friend void from_json(nlohmann::json const &j, Topo &topo) {
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

                        m.delta_E =
                            (m.count * m.delta_E + delta_E) / (m.count + 1);

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

    ////////////////////////////////////////////////////////////

    using Key_t = typename Canon::Key_t;

    std::unordered_map<Key_t, Topo> catalog;

    static constexpr auto dirname = "dump";

  public:
    Catalog() {
        using nlohmann::json;
        std::string keys = std::string{dirname} + "/rdf.json";

        if (fileExist(keys)) {
            json rdfs = json::parse(std::ifstream(keys));

            for (auto &&elem : rdfs) {
                Key_t r = elem.get<Key_t>();

                auto fname = std::string{dirname} + '/' + r.to_string();
                check(fileExist(fname), "missing a topo file");

                json j = json::parse(std::ifstream(fname));
                catalog[r] = j.get<Topo>();
            }
        } else {
            std::cout << "Missing " << keys << std::endl;
        }
    }

    inline auto &operator[](Key_t const &k) { return catalog[k].getMechs(); }

    inline auto const &operator[](Key_t const &k) const {
        return catalog[k].getMechs();
    }

    void write() const {
        using nlohmann::json;
        json keys;
        std::unordered_set<std::string> names{};

        for (auto &&[rdf, topo] : catalog) {
            json j = topo;

            std::string name = std::string{dirname} + '/' + rdf.to_string();
            check(names.count(name) == 0, "need better naming");

            std::ofstream(name) << j.dump(2);

            keys.push_back(rdf);

            names.insert(std::move(name));
        }
        std::ofstream(std::string{dirname} + "/rdf.json") << keys.dump(2);
    }

    template <typename F, typename C, typename MinImage>
    void update(Vector const &x, F const &f, C const &cl, MinImage const &mi) {

        using result_t = std::vector<std::tuple<Vector, Vector>>;

        std::vector<result_t> searches;

        for (std::size_t i = 0; i < cl.size(); ++i) {

            catalog[cl[i]].count += 1; // default constructs new

            constexpr std::size_t sp_trys = 5;

            while (catalog[cl[i]].sp_searches < 25 ||
                   catalog[cl[i]].sp_searches <
                       std::sqrt(catalog[cl[i]].count)) {

                catalog[cl[i]].sp_searches += sp_trys;

                std::cout << "@ " << i << '/' << cl.size()
                          << " dimer launch ... " << std::flush;

                searches.push_back(findSaddle(sp_trys, x, i, f, mi));

                std::cout << searches.back().size() << " successful!"
                          << std::endl;
            }
        }

        double f_x = f(x);

        for (auto &&vec : searches) {
            for (auto &&[sp, end] : vec) {

                auto [centre, ref] = cl.classifyMech(end);

                catalog[cl[centre]].pushMech(f(sp) - f_x, f(end) - f_x,
                                             std::move(ref));
            }
        }
    }
};
