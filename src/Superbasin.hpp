#pragma once

#include <list>
#include <random>
#include <vector>

#include "pcg_random.hpp"

#include "Catalog.hpp"
#include "utils.hpp"

inline constexpr double ARRHENIUS_PRE = 5.12e12;                 // Hz
inline constexpr double TEMP = 300;                              // k
inline constexpr double KB_T = 1380649.0 / 16021766340.0 * TEMP; // eV K^-1
inline constexpr double INV_KB_T = 1 / KB_T;                     // eV^-1 K

inline constexpr double SUPER_BASIN_TOL = 0.4;           // eV
inline constexpr std::size_t SUPER_BASIN_MIN_SIZE = 2;   // states
inline constexpr std::size_t SUPER_BASIN_MAX_SIZE = 256; // states
inline constexpr std::size_t SUPER_BASIN_CACHE_SIZE = 8; // superbasins

inline constexpr double SUPER_BASIN_ACTIVE_HARDCAP = 2.5; // eV

double toRate(double active_E) {
    CHECK(active_E > 0, "sp energy < init energy " << active_E);
    return ARRHENIUS_PRE * std::exp(active_E * -INV_KB_T);
}

// POD representation of generalised mech acting at a specific atom in the
// lattice.
struct LocalMech {
    std::size_t atom_idx;
    std::size_t mech_idx;
    double rate;
    bool low_barrier;      // delta_E fwd & rev are small
    bool exit_mech = true; // links: inside sb -> outside sb
};

// POD represnetation of a single basin: combination of a state and all possible
// mechanism leading out of mechanism.
struct Basin {
    Vector state;
    double r_sum{};
    double inv_sum{};
    std::vector<LocalMech> mechs{};

    template <typename T> Basin(Catalog<T> const &catalog, Vector const &x);
};

template <typename T>
Basin::Basin(Catalog<T> const &catalog, Vector const &x) : state{x} {

    for (std::size_t i = 0; i < catalog.size(); ++i) {
        Topology const &t = catalog[i];

        for (std::size_t j = 0; j < t.mechs.size(); ++j) {
            double const fwd = t.mechs[j].active_E;

            if (fwd < SUPER_BASIN_ACTIVE_HARDCAP) {
                double const rev = fwd - t.mechs[j].delta_E;
                double const rate = toRate(fwd);

                mechs.push_back(
                    {i, j, rate,
                     fwd < SUPER_BASIN_TOL && rev < SUPER_BASIN_TOL});

                r_sum += rate;
            }
        }
    }

    inv_sum = 1 / r_sum;
}

// Manages a superbasin: a collection of low barrier linked basins.
class SuperBasin {
  public:
    void connect(std::size_t i, std::size_t m, std::size_t j);
    std::size_t expand_and_occupy(Basin &&basin);
    std::optional<std::size_t> find_and_occupy(Vector const &state);

    Eigen::VectorXd tau() const;

    std::size_t occupied() const { return occupied_basin; }
    std::size_t size() const { return super.size(); }

    Basin const &operator[](std::size_t i) const { return super[i]; }

  private:
    auto theta() const;

    std::size_t occupied_basin{}; // Index of current basin in super;
    std::vector<Basin> super{};   // Collection of basins
    Eigen::MatrixXd t_prob{};     // Transition priobability matrix;
};

// Non-allocating eigen object theta_{i} = kroneker_{is}
auto SuperBasin::theta() const {
    return Eigen::VectorXd::NullaryExpr(
        size(), [s = occupied_basin](std::size_t i) { return i == s; });
}

// Compute vector of mean residence-times in each basin
Eigen::VectorXd SuperBasin::tau() const {

    auto const identity = Eigen::MatrixXd::Identity(size(), size());

    // calc theta^{sum}
    Eigen::VectorXd tau =
        (identity - t_prob).colPivHouseholderQr().solve(theta());

    // convert to tau_i
    for (std::size_t i = 0; i < super.size(); ++i) {
        tau[i] *= super[i].inv_sum;
    }

    return tau;
}

// Adds basin to superbasin and returns index of location, sets new superbasin
// as the occupied basin
std::size_t SuperBasin::expand_and_occupy(Basin &&basin) {
    super.push_back(std::move(basin));
    t_prob.conservativeResizeLike(Eigen::MatrixXd::Zero(size(), size()));
    return (occupied_basin = size() - 1);
}

// Connect internal state i via mechanism m to internal state j
void SuperBasin::connect(std::size_t i, std::size_t m, std::size_t j) {
    t_prob(j, i) = super[i].mechs[m].rate * super[i].inv_sum;
    super[i].mechs[m].exit_mech = false;
    return;
}

//  Return index of basin corresponding to "state" and set basin as occupied
//  basin.
std::optional<std::size_t> SuperBasin::find_and_occupy(Vector const &state) {
    for (std::size_t i = 0; i < size(); ++i) {
        if (((state - super[i].state).abs() < DIST_TOL).all()) {
            occupied_basin = i;
            return i;
        }
    }
    return std::nullopt;
}

// Manages a collection of superbasins (active vs cached) and implements
// super KMC and standard KMC algorithms to select next state returning the time
// increment and mechanism. Assumes the selected mechanism is applied via
// catalog.reconstruct() into "x" between calls to .chooseMechanism()
class KineticMC {
  public:
    template <typename T>
    std::tuple<double, std::size_t, Mechanism const &>
    chooseMechanism(Catalog<T> &catalog, Vector &x);

    KineticMC()
        : overflow_flag{false},
          rng(pcg_extras::seed_seq_from<std::random_device>{}),
          uniform_dist(0, 1) {}

  private:
    struct Pair {
        std::size_t basin;
        std::size_t mech;
    };

    template <typename T>
    std::pair<Pair, double> dispatchKMC(Catalog<T> &catalog, Vector &x);

    std::pair<Pair, double> standardKMC(std::size_t basin);
    std::pair<Pair, double> superKMC();

    bool try_cache(SuperBasin &&basin);

    std::list<SuperBasin> cache; // old superbasins
    bool overflow_flag;          // signals superbasin grown too big
    SuperBasin sb;               // active superbasin
    Pair prev;                   // previous basin + escape_mech

    pcg64 rng;
    std::uniform_real_distribution<double> uniform_dist;
};

template <typename T>
std::tuple<double, std::size_t, Mechanism const &>
KineticMC::chooseMechanism(Catalog<T> &catalog, Vector &x) {

    auto const [choice, r_sum] = dispatchKMC(catalog, x);

    if (!overflow_flag) {
        std::cout << "Super B: " << sb.occupied() << '/' << sb.size() << '\n';
    } else {
        std::cout << "Super B: overflow\n";
    }

    prev = choice;

    // May have chosen mech from different state
    if (choice.basin != sb.occupied()) {
        x = sb[choice.basin].state;
        catalog.analyzeTopology(x);
    }

    std::size_t const atom_idx = sb[choice.basin].mechs[choice.mech].atom_idx;
    std::size_t const mech_idx = sb[choice.basin].mechs[choice.mech].mech_idx;

    CHECK(((x - sb[choice.basin].state).abs() < DIST_TOL).all(),
          "SHOULD match");

    CHECK(dot(x - sb[choice.basin].state, x - sb[choice.basin].state) < 0.1,
          "DIST_TOL not strong enough");

    return {-std::log(uniform_dist(rng)) / r_sum, atom_idx,
            catalog[atom_idx].mechs[mech_idx]};
}

// Organise internal state of KineticMC object (fetch cache, update superbasin
// linkage, etc) then call appropriate KMC routine to get a mechanism choice.
template <typename T>
std::pair<KineticMC::Pair, double> KineticMC::dispatchKMC(Catalog<T> &catalog,
                                                          Vector &x) {

    if (sb.size() == 0) {
        // If empty superbasin, do normal KMC
        return standardKMC(sb.expand_and_occupy(Basin{catalog, x}));
    } else if (sb.size() > SUPER_BASIN_MAX_SIZE) {
        // Either loaded cached, overflowing sb or expanded beyond max size
        overflow_flag = true;
    }

    if (overflow_flag) {
        if (sb[prev.basin].mechs[prev.mech].low_barrier) {
            try_cache(std::exchange(sb, {}));
            return standardKMC(sb.expand_and_occupy(Basin{catalog, x}));
        } else {
            overflow_flag = false;
        }
    } else {
        // Try and connect states if did internal jump
        if (std::optional<std::size_t> basin = sb.find_and_occupy(x)) {
            sb.connect(prev.basin, prev.mech, *basin);
            return superKMC();
        }
        // Sb needs expand_and_occupying if followed low barrier
        if (sb[prev.basin].mechs[prev.mech].low_barrier) {
            std::size_t j = sb.expand_and_occupy(Basin{catalog, x});
            sb.connect(prev.basin, prev.mech, j);
            return standardKMC(j);
        }
    }

    // Therefore followed high barrir out of basin

    // Try and retrive cached sb
    auto cached = [&]() -> std::optional<SuperBasin> {
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (std::optional<std::size_t> basin = it->find_and_occupy(x)) {
                SuperBasin tmp = std::move(*it);
                cache.erase(it);
                return tmp;
            }
        }
        return std::nullopt;
    }();

    if (cached) {
        std::cout << "!===========LOAD_CACHED===========!\n" << std::endl;
        try_cache(std::exchange(sb, std::move(*cached)));
        return superKMC();
    } else {
        std::cout << "!============NEW_SUPER============!\n" << std::endl;
        try_cache(std::exchange(sb, {}));
        return standardKMC(sb.expand_and_occupy(Basin{catalog, x}));
    }
}

// Choose a mechanism from basin "basin" in sb using the standard kmc
// algorithm, return the choice as a pair and the total rate of the choice
std::pair<KineticMC::Pair, double> KineticMC::standardKMC(std::size_t basin) {

    std::size_t const choice = [&]() {
        double const lim = uniform_dist(rng) * sb[basin].r_sum;
        double sum = 0;

        for (std::size_t i = 0; i < sb[basin].mechs.size(); ++i) {
            if (sb[basin].mechs[i].exit_mech) {
                sum += sb[basin].mechs[i].rate;
                if (sum > lim) {
                    return i;
                }
            }
        }
        throw std::runtime_error("Hit end of normal choice");
    }();

    double const rate = sb[basin].mechs[choice].rate;

    std::cout << "Rate:    " << rate << " : " << rate / sb[basin].r_sum * 100
              << "%\n";

    return {{basin, choice}, sb[basin].r_sum};
}

// Choose a mechanism from any basin in sb using the superbasin kmc
// algorithm, return the choice as a pair{basin, mech} and the total rate of the
// choice
std::pair<KineticMC::Pair, double> KineticMC::superKMC() {

    Eigen::VectorXd tau = sb.tau();

    double const inv_tau = 1 / tau.sum();

    // Sum over all basin->escape rate times basin modifiers, omit normalising
    // factor of 1/tau
    double const r_sum = [&]() {
        double sum = 0;

        for (std::size_t i = 0; i < sb.size(); ++i) {
            double basin_sum = 0;
            for (std::size_t j = 0; j < sb[i].mechs.size(); ++j) {
                // basin -> non-basin
                if (sb[i].mechs[j].exit_mech) {
                    basin_sum += sb[i].mechs[j].rate;
                }
            }
            sum += tau[i] * basin_sum;
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
                    sum += tau[i] * sb[i].mechs[j].rate;
                    if (sum > lim) {
                        return {i, j};
                    }
                }
            }
        }
        throw std::runtime_error("Hit end of super choice");
    }();

    double const eff_rate =
        tau[choice.basin] * inv_tau * sb[choice.basin].mechs[choice.mech].rate;

    double const prob = 100 * (eff_rate / inv_tau) / r_sum;

    std::cout << "Effrate: " << eff_rate << " : " << prob << "%\n";

    // must normalize by inv_tau
    return {choice, r_sum * inv_tau};
}

// If sb large enough cache it and delte old sbs if nessasary
bool KineticMC::try_cache(SuperBasin &&basin) {
    if (basin.size() >= SUPER_BASIN_MIN_SIZE) {
        cache.push_front(std::move(basin));

        if (cache.size() > SUPER_BASIN_CACHE_SIZE) {
            cache.pop_back();
        }
        return true;
    }
    return false;
}
