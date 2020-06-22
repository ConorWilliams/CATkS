#define NDEBUG
#define EIGEN_NO_DEBUG

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include <future>
#include <thread>

#include "pcg_random.hpp"

#include "utils.hpp"

#include "Dimer.hpp"
#include "Forces.hpp"
#include "Minimise.hpp"

#include "Classes.hpp"
#include "Colour.hpp"
#include "DumpXYX.hpp"
#include "Rdf.hpp"

// controls displacement along mm at saddle
inline constexpr double NUDGE = 0.1;
// tollerence for 3N vectors to be considered the same vector
inline constexpr double TOL_NEAR = 0.1;

constexpr double G_SPHERE = 4;
constexpr double G_AMP = 0.325;

// init is a minimised (unporturbed) vector of atoms
// idx is centre of displacemnet
// num is number of atempts
// f is their force object
template <typename T>
std::vector<std::pair<Vector, Vector>>
findSaddle(Vector const &init, std::size_t idx, std::size_t num, T const &f) {

    // Centre atom coordinates
    double x = init[idx * 3 + 0];
    double y = init[idx * 3 + 1];
    double z = init[idx * 3 + 2];

    // Seed with a real random value, if available
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::normal_distribution<> gauss_dist(0, G_AMP);

    Vector sp{init.size()};
    Vector ax{init.size()};

    Dimer dimer{f, sp, ax, [&]() { /*output(sp, col); */ }};

    Minimise min{f, f, init.size()};

    std::vector<std::pair<Vector, Vector>> saddle_points;

    for (std::size_t anon = 0; anon < num; ++anon) {
        for (int i = 0; i < init.size(); i = i + 3) {
            double dist = f.periodicNormSq(x, y, z, init[i + 0], init[i + 1],
                                           init[i + 2]);

            if (dist < G_SPHERE * G_SPHERE) {
                sp[i + 0] = init[i + 0] + gauss_dist(rng);
                sp[i + 1] = init[i + 1] + gauss_dist(rng);
                sp[i + 2] = init[i + 2] + gauss_dist(rng);

                ax[i + 0] = gauss_dist(rng);
                ax[i + 1] = gauss_dist(rng);
                ax[i + 2] = gauss_dist(rng);
            } else {
                sp[i + 0] = init[i + 0];
                sp[i + 1] = init[i + 1];
                sp[i + 2] = init[i + 2];

                ax[i + 0] = 0;
                ax[i + 1] = 0;
                ax[i + 2] = 0;
            }
        }

        ax.matrix().normalize();

        if (!dimer.findSaddle()) {
            // failed SP search
            continue;
        }

        Vector old = sp + ax * NUDGE;
        Vector end = sp - ax * NUDGE;

        if (!min.findMin(old) || !min.findMin(end)) {
            // failed minimisation
            continue;
        }

        double distOld = dot(old - init, old - init);
        double distFwd = dot(end - init, end - init);

        // want old to be init
        if (distOld > distFwd) {
            using std::swap;
            swap(old, end);
            swap(distOld, distFwd);
        }

        if (distOld > TOL_NEAR) {
            // disconnected SP
            continue;
        }

        if (dot(end - old, end - old) < TOL_NEAR) {
            // minimasations both converged to init
            continue;
        }

        saddle_points.emplace_back(sp, end);
    }

    return saddle_points;
}

inline constexpr double ARRHENIUS_PRE = 1e13;
inline constexpr double KB_T = 8.617333262145 * 1e-5 * 500; // eV K^-1

inline constexpr double INV_KB_T = 1 / KB_T;

double activeToRate(double active_E) {

    check(active_E > 0, "sp energy < init energy");

    return ARRHENIUS_PRE * std::exp(-active_E * INV_KB_T);
}

//
template <typename T>
auto updateCatalog(std::unordered_map<Rdf, Topology> &catalog, Vector const &x,
                   T const &f) {

    std::vector<Rdf> topos = f.colourAll(x);

    using sp_vec_t = std::vector<std::pair<Vector, Vector>>;

    std::vector<std::future<sp_vec_t>> saddle_points;

    for (std::size_t i = 0; i < topos.size(); ++i) {

        if (auto search = catalog.find(topos[i]); search != catalog.end()) {
            // discovered topo before
            ++(search->second.count);
        } else {
            // new topology
            catalog[topos[i]].count += 1;
        }

        std::size_t count = catalog[topos[i]].count;

        constexpr std::size_t try_n = 5;

        while (catalog[topos[i]].sp_searches < 50 ||
               catalog[topos[i]].sp_searches < std::sqrt(count)) {

            catalog[topos[i]].sp_searches += try_n;

            std::cout << "@ " << i << " Dimer launch ... " << std::flush;

            constexpr auto findSaddleHelp = [](auto &&... args) -> sp_vec_t {
                return findSaddle(std::forward<decltype(args)>(args)...);
            };

            saddle_points.push_back(
                std::async(std::launch::async, findSaddleHelp, x, i, try_n, f));

            std::cout << "found " << saddle_points.size() << std::endl;
        }
    }

    for (auto &&sp_vec : saddle_points) {
        for (auto &&[sp, end] : sp_vec.get()) {

            auto [centre, ref] = classifyMech(x, end, f);

            catalog[topos[centre]].pushMech(f(sp) - f(x), f(end) - f(x),
                                            std::move(ref));
        }
    }

    return topos;
}

template <typename T>
void outputAllMechs(std::unordered_map<Rdf, Topology> &catalog, Vector const &x,
                    T const &f) {
    std::vector<Rdf> topos = f.colourAll(x);

    std::unordered_set<Rdf> done{};

    std::cout << "writing all mechanisms" << std::endl;

    output(x);

    for (std::size_t i = 0; i < topos.size(); ++i) {
        if (done.count(topos[i]) == 0) {
            done.insert(topos[i]);

            for (auto &&m : catalog[topos[i]].getMechs()) {
                std::cout << FRAME << ' ' << m.active_E << ' ' << m.delta_E
                          << std::endl;
                output(reconstruct(x, i, m.ref, f));
            }
        }
    }
}

enum : uint8_t { Fe = 0, H = 1 };
constexpr double LAT = 2.855700;

inline constexpr int len = 5;

static const std::string OUTFILE = "/home/cdt1902/dis/CATkS/raw.txt";

struct LocalisedMech {
    std::size_t atom;
    double rate;
    std::vector<Eigen::Vector3d> const &ref;
};

int main() {

    Vector init(len * len * len * 3 * 2 - 3);
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if ((i == 1 && j == 1 && k == 1) /*||
                    (i == 4 && j == 1 && k == 1)*/) {
                    init[3 * cell + 0] = (i + 0.5) * LAT;
                    init[3 * cell + 1] = (j + 0.5) * LAT;
                    init[3 * cell + 2] = (k + 0.5) * LAT;

                    cell += 1;

                } else {
                    init[3 * cell + 0] = i * LAT;
                    init[3 * cell + 1] = j * LAT;
                    init[3 * cell + 2] = k * LAT;

                    init[3 * cell + 3] = (i + 0.5) * LAT;
                    init[3 * cell + 4] = (j + 0.5) * LAT;
                    init[3 * cell + 5] = (k + 0.5) * LAT;

                    cell += 2;
                }
            }
        }
    }

    // kinds.back() = H;
    //
    // init[init.size() - 3] = LAT;
    // init[init.size() - 2] = LAT;
    // init[init.size() - 1] = 0.5 * LAT;

    FuncEAM f{
        "/home/cdt1902/dis/CATkS/data/PotentialA.fs",
        kinds,
        0,
        len * LAT,
        0,
        len * LAT,
        0,
        len * LAT,
    };

    f.sort(init);

    Minimise min{f, f, init.size()};

    min.findMin(init);

    std::unordered_map<Rdf, Topology> catalog; // = readMap("dump");

    double time = 0;

    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::uniform_real_distribution<> uniform_dist(0, 1);

    for (int anon = 0; anon < 50; ++anon) {

        // if (anon == 0) {
        //     outputAllMechs(catalog, init, f);
        //     return 0;
        // }

        output(init, f.quasiColourAll(init));

        std::vector<Rdf> topos = updateCatalog(catalog, init, f);

        // writeMap("dump", catalog);

        std::vector<LocalisedMech> possible{};

        for (std::size_t i = 0; i < topos.size(); ++i) {
            for (auto &&m : catalog[topos[i]].getMechs()) {
                possible.push_back({i, activeToRate(m.active_E), m.ref});
            }
        }

        double rate_sum = std::accumulate(
            possible.begin(), possible.end(), 0.0,
            [](double sum, LocalisedMech const &m) { return sum + m.rate; });

        double p1 = uniform_dist(rng);

        LocalisedMech choice = [&]() {
            double sum = 0;
            for (auto &&elem : possible) {
                sum += elem.rate;
                if (sum > p1 * rate_sum) {
                    return elem;
                }
            }
            std::terminate();
        }();

        time += -std::log(uniform_dist(rng)) / rate_sum;

        init = reconstruct(init, choice.atom, choice.ref, f);

        // min.findMin(init);

        std::cout << "TIME: " << time << std::endl;

        // f.sort(init);

        ////////////////////OLD Reconstruct test////////////////////////////

        // auto [err, sp, end] = findSaddle(init, 12, f);
        //
        // if (!err) {
        //     output(end);
        //
        //     auto [centre, ref] = classifyMech(init, end, f);
        //
        //     std::cout << "\nCentre was " << centre << std::endl;
        //
        //     Vector recon = reconstruct(init, 12, ref, f);
        //
        //     std::cout << f(end) - f(init) << std::endl;
        //     std::cout << f(recon) - f(init) << std::endl;
        //
        //     min.findMin(recon);
        //
        //     output(recon);
        //
        //     return 0;
        // }

        //////////////////////////////////////////////////////
    }
}
