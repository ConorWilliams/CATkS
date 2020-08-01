#pragma once

#include <list>
#include <random>
#include <vector>

#include "MurmurHash3.h"
#include "pcg_random.hpp"

#include "Catalog.hpp"
#include "Cbuff.hpp"
#include "utils.hpp"

inline constexpr double ARRHENIUS_PRE = 5.12e12;
inline constexpr double TEMP = 300;                              // k
inline constexpr double KB_T = 1380649.0 / 16021766340.0 * TEMP; // eV K^-1
inline constexpr double INV_KB_T = 1 / KB_T;

inline constexpr double SUPER_BASIN_TOL = 0.1;          // eV
inline constexpr std::size_t SUPER_BASIN_MIN_SIZE = 2;  // states
inline constexpr std::size_t SUPER_BASIN_MAX_SIZE = 25; // states
inline constexpr std::size_t SUPER_BASIN_CACHE_SIZE = 5;

using Hash_t = std::array<std::uint64_t, 2>;

double toRate(double active_E) {

    CHECK(active_E > 0, "sp energy < init energy " << active_E);

    return ARRHENIUS_PRE * std::exp(active_E * -INV_KB_T);
}

struct LocalMech {
    std::size_t atom_idx;
    std::size_t mech_idx;
    double rate;
    bool low_barrier;
    bool exit_mech = true;
};

Hash_t hash_state(Vector const &x) {
    std::vector<int> tmp = transform_into(
        x.begin(), x.end(), [](double x) -> int { return x / DIST_TOL; });

    Hash_t hash{};
    MurmurHash3_x86_128(tmp.data(), sizeof(int) * tmp.size(), 0, hash.data());
    std::cout << hash[0] << hash[1] << '\n';
    return hash;
}

struct Basin {
    Vector state;
    double inv_sum{};
    std::vector<LocalMech> mechs{};
};

template <typename T>
Basin construct_basin(Catalog<T> const &catalog, Vector const &x) {

    Basin basin{x};

    double sum = 0;

    for (std::size_t i = 0; i < catalog.size(); ++i) {
        Topology const &t = catalog[i];

        for (std::size_t j = 0; j < t.mechs.size(); ++j) {
            double const fwd = t.mechs[j].active_E;
            double const rev = fwd - t.mechs[j].delta_E;

            double const rate = toRate(fwd);

            basin.mechs.push_back(
                {i, j, rate, fwd < SUPER_BASIN_TOL && rev < SUPER_BASIN_TOL});

            sum += rate;
        }
    }

    basin.inv_sum = 1 / sum;

    return basin;
}

class SuperBasin {
  public:
    std::size_t expand(Basin &&basin);
    void connect(std::size_t i, std::size_t m, std::size_t j);

    std::optional<std::size_t> find(Vector const &state) const;
    Eigen::VectorXd tau() const;

    std::size_t size() const { return super.size(); }
    Basin const &operator[](std::size_t i) { return super[i]; }

    std::size_t occupied_basin{}; // Index of current basin in super;
  private:
    std::vector<Basin> super{}; // Collection
    Eigen::MatrixXd t_prob{};   // Transition priobability matrix;
};

Eigen::VectorXd SuperBasin::tau() const {
    // theta_0 def
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(size());
    theta[occupied_basin] = 1;

    // calc theta^{sum}
    theta = (Eigen::MatrixXd::Identity(size(), size()) - t_prob)
                .colPivHouseholderQr()
                .solve(theta);

    // convert to tau_i
    for (std::size_t i = 0; i < super.size(); ++i) {
        theta[i] *= super[i].inv_sum;
    }

    return theta;
}

// adds basin to superbasin and returns index of location
std::size_t SuperBasin::expand(Basin &&basin) {

    super.push_back(std::move(basin));

    t_prob.conservativeResizeLike(Eigen::MatrixXd::Zero(size(), size()));

    occupied_basin = size() - 1;

    return occupied_basin;
}

// Connect internal state i via mechanism m to internal state j.
void SuperBasin::connect(std::size_t i, std::size_t m, std::size_t j) {

    t_prob(j, i) = super[i].mechs[m].rate * super[i].inv_sum;

    super[i].mechs[m].exit_mech = false;
}

std::optional<std::size_t> SuperBasin::find(Vector const &state) const {
    for (std::size_t i = 0; i < size(); ++i) {
        if (((state - super[i].state).abs() < DIST_TOL).all()) {
            return i;
        }
    }
    return std::nullopt;
}

class KineticMC {
  public:
    template <typename T>
    std::pair<double, double> advanceState(Catalog<T> &catalog, Vector &x);

    KineticMC()
        : rng(pcg_extras::seed_seq_from<std::random_device>{}),
          uniform_dist(0, 1) {}

  private:
    struct Pair {
        std::size_t basin;
        std::size_t mech;
    };

    std::list<SuperBasin> cache; // old superbasins
    SuperBasin sb;               // active superbasin
    Pair prev;                   // previous basin + escape_mech

    pcg64 rng;
    std::uniform_real_distribution<double> uniform_dist;

    void try_cache(SuperBasin &&basin);

    template <typename T>
    std::pair<Pair, double> dispatchKMC(Catalog<T> &catalog, Vector &x);

    std::pair<Pair, double> standardKMC(std::size_t basin);
    std::pair<Pair, double> superKMC();
};

void KineticMC::try_cache(SuperBasin &&basin) {
    if (basin.size() >= SUPER_BASIN_MIN_SIZE) {
        cache.push_front(std::move(basin));

        if (cache.size() > SUPER_BASIN_CACHE_SIZE) {
            cache.pop_back();
        }
    }
}

template <typename T>
std::pair<KineticMC::Pair, double> KineticMC::dispatchKMC(Catalog<T> &catalog,
                                                          Vector &x) {
    if (sb.size() == 0) {
        // Empty superbasin, do normal KMC
        // std::cout << "first timer\n";
        return standardKMC(sb.expand(construct_basin(catalog, x)));
    }

    // Try and connect states if did internal jump
    if (std::optional<std::size_t> basin = sb.find(x)) {
        // std::cout << "internal jump\n";
        sb.connect(prev.basin, prev.mech, *basin);
        sb.occupied_basin = *basin;
        return superKMC();
    }

    // Sb needs expanding if followed low barrier
    if (sb[prev.basin].mechs[prev.mech].low_barrier) {
        // std::cout << "expanding sb\n";
        std::size_t j = sb.expand(construct_basin(catalog, x));
        sb.connect(prev.basin, prev.mech, j);
        return standardKMC(j);
    }

    std::cout << "Escape Artist!\n";

    // sb = SuperBasin{};
    // return standardKMC(sb.expand(construct_basin(catalog, x)));

    auto cached = [&]() -> std::optional<SuperBasin> {
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (std::optional<std::size_t> basin = it->find(x)) {

                it->occupied_basin = *basin;

                SuperBasin tmp = std::move(*it);

                // x = tmp[*basin].state;
                // catalog.analyzeTopology(x);

                cache.erase(it);
                return tmp;
            }
        }
        return std::nullopt;
    }();

    if (cached) {
        // std::cout << "LOAD CACHE" << std::endl;
        try_cache(std::exchange(sb, std::move(*cached)));
        return superKMC();
    } else {
        // std::cout << "FRESH SB" << std::endl;
        try_cache(std::exchange(sb, {}));
        return standardKMC(sb.expand(construct_basin(catalog, x)));
    }
}

template <typename T>
std::pair<double, double> KineticMC::advanceState(Catalog<T> &catalog,
                                                  Vector &x) {

    auto const [choice, r_sum] = dispatchKMC(catalog, x);

    std::cout << "Superbasin " << sb.occupied_basin << '/' << sb.size() << '\n';

    prev = choice;

    // May have chosen mech from different state
    if (choice.basin != sb.occupied_basin) {
        x = sb[choice.basin].state;
        catalog.analyzeTopology(x);
    }

    std::size_t const atom_idx = sb[choice.basin].mechs[choice.mech].atom_idx;
    std::size_t const mech_idx = sb[choice.basin].mechs[choice.mech].mech_idx;

    CHECK(((x - sb[choice.basin].state).abs() < DIST_TOL).all(),
          "SHOULD match");

    Mechanism const &m = catalog[atom_idx].mechs[mech_idx];

    std::cout << "Barrier: " << m.active_E << '\n';

    catalog.reconstruct(atom_idx, x, m);

    return {-std::log(uniform_dist(rng)) / r_sum, m.delta_E};
}

// Choose a mechanism from basin "basin" in sb using the standard kmc
// algorithm, return the choice as a pair and the total rate of the choice
std::pair<KineticMC::Pair, double> KineticMC::standardKMC(std::size_t basin) {
    double const r_sum = [&]() {
        double sum = 0;

        for (std::size_t i = 0; i < sb[basin].mechs.size(); ++i) {
            if (sb[basin].mechs[i].exit_mech) {
                // basin -> non-basin
                sum += sb[basin].mechs[i].rate;
            }
        }

        return sum;
    }();

    std::size_t const choice = [&]() {
        double const lim = uniform_dist(rng) * r_sum;
        double sum = 0;

        for (std::size_t i = 0; i < sb[basin].mechs.size(); ++i) {
            if (sb[basin].mechs[i].exit_mech) {
                // basin -> non-basin
                sum += sb[basin].mechs[i].rate;
                if (sum > lim) {
                    return i;
                }
            }
        }

        std::cout << "Hit end of choice\n";
        std::terminate();
    }();

    double const rate = sb[basin].mechs[choice].rate;

    std::cout << "Rate:    " << rate << " : " << rate / r_sum << '\n';

    return {{basin, choice}, r_sum};
}

// Choose a mechanism from any basin in sb using the superbasin kmc
// algorithm, return the choice as a pair and the total rate of the choice
std::pair<KineticMC::Pair, double> KineticMC::superKMC() {
    // theta_0 def
    Eigen::VectorXd tau = sb.tau();

    double const inv_tau = 1 / tau.sum();

    double const r_sum = [&]() {
        double sum = 0;

        for (std::size_t i = 0; i < sb.size(); ++i) {
            for (std::size_t j = 0; j < sb[i].mechs.size(); ++j) {
                if (sb[i].mechs[j].exit_mech) {
                    // basin -> non-basin
                    sum += tau[i] * inv_tau * sb[i].mechs[j].rate;
                }
            }
        }
        return sum;
    }();

    CHECK(r_sum > 0, "o no sumin wong");

    Pair const choice = [&]() -> Pair {
        double const lim = uniform_dist(rng) * r_sum;
        double sum = 0;

        for (std::size_t i = 0; i < sb.size(); ++i) {
            for (std::size_t j = 0; j < sb[i].mechs.size(); ++j) {
                if (sb[i].mechs[j].exit_mech) {
                    sum += tau[i] * inv_tau * sb[i].mechs[j].rate;
                    if (sum > lim) {
                        return {i, j};
                    }
                }
            }
        }

        std::cout << "Hit end of choice\n";
        std::terminate();
    }();

    double const eff_rate =
        tau[choice.basin] * inv_tau * sb[choice.basin].mechs[choice.mech].rate;

    std::cout << "Effrate: " << eff_rate << " : " << eff_rate / r_sum << '\n';

    return {choice, r_sum};
}
