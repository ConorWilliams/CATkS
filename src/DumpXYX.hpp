#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "nlohmann/json.hpp"

#include <utils.hpp>

template <typename K>
void dumpXYX(std::string const &file, Vector const &coords, K kinds) {
    check(kinds.size() * 3 == (std::size_t)coords.size(),
          " wrong number of atoms/kinds");

    std::ofstream outfile{file};

    outfile << kinds.size() << std::endl;
    outfile << "This is C.J.Williams' dumpfile";

    for (std::size_t i = 0; i < kinds.size(); ++i) {
        outfile << '\n'
                << kinds[i] << ' ' << coords[3 * i + 0] << ' '
                << coords[3 * i + 1] << ' ' << coords[3 * i + 2];
    }
}

inline constexpr double RATE_TOL = 0.01;
inline constexpr double DELTA_E_TOL = 0.01;
inline constexpr double DIST_TOL = 0.1;

class Topology {
  private:
    struct Mech {
        double rate;
        double delta_E;
        std::vector<Eigen::Vector3d> ref;

        Mech(double rate, double delta_E, std::vector<Eigen::Vector3d> &&ref)
            : rate{rate}, delta_E{delta_E}, ref{std::move(ref)} {}
    };

    std::vector<Mech> mechs{}; // Mechs for this topology

  public:
    std::size_t count = 1;       // Num times occured in the simulation
    std::size_t sp_searches = 0; // Total sp searches initated from topology

    void pushMech(double rate, double delta_E,
                  std::vector<Eigen::Vector3d> &&ref) {

        constexpr auto norm_comp = [](Eigen::Vector3d const &a,
                                      Eigen::Vector3d const &b) -> bool {
            return a.squaredNorm() < b.squaredNorm();
        };

        Eigen::Vector3d const key =
            *std::max_element(ref.begin(), ref.end(), norm_comp);

        for (auto &&m : mechs) {
            if (std::abs(m.rate - rate) > RATE_TOL ||
                std::abs(m.delta_E - delta_E) > DELTA_E_TOL) {

                Eigen::Vector3d m_key =
                    *std::max_element(m.ref.begin(), m.ref.end(), norm_comp);

                m_key = (m_key - key).array().abs();

                if ((m_key > DIST_TOL).array().any()) {
                }
            }
        }

        mechs.emplace_back(rate, delta_E, std::move(ref));
    }
};
