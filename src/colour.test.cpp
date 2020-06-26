#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "pcg_random.hpp"

#include "Classes.hpp"

#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Minimise.hpp"
#include "Rdf.hpp"
#include "Topo.hpp"
#include "utils.hpp"

// controls displacement along mm at saddle
inline constexpr double NUDGE = 0.1;
// tollerence for 3N vectors to be considered the same vector
inline constexpr double TOL_NEAR = 0.1;

constexpr double G_SPHERE = 4;
constexpr double G_AMP = 0.325;

// init is a minimised (unporturbed) vector of atoms
// idx is centre of displacemnet
// f is their force object
template <typename T>
std::tuple<int, Vector, Vector> findSaddle(Vector const &init, std::size_t idx,
                                           T const &f) {
    Vector sp = init;
    Vector ax = Vector::Zero(init.size());

    // random pertabations

    // Seed with a real random value, if available
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::normal_distribution<> gauss_dist(0, G_AMP);

    double x = init[idx * 3 + 0];
    double y = init[idx * 3 + 1];
    double z = init[idx * 3 + 2];

    std::vector<int> col(init.size() / 3, 0);

    for (int i = 0; i < init.size(); i = i + 3) {
        double dist =
            f.periodicNormSq(x, y, z, init[i + 0], init[i + 1], init[i + 2]);

        if (dist < G_SPHERE * G_SPHERE) {
            sp[i + 0] += gauss_dist(rng);
            sp[i + 1] += gauss_dist(rng);
            sp[i + 2] += gauss_dist(rng);

            ax[i + 0] = gauss_dist(rng);
            ax[i + 1] = gauss_dist(rng);
            ax[i + 2] = gauss_dist(rng);

            col[i / 3] = 1;
        }
    }

    // ax.matrix().normalize(); // not strictly nessaserry, done in ctr

    // output(sp, col); // tmp

    Dimer dimer{f, sp, ax, [&]() { /*output(sp, col); */ }};

    if (!dimer.findSaddle()) {
        // failed SP search
        return {1, Vector{}, Vector{}};
    }

    Vector old = sp + ax * NUDGE;
    Vector end = sp - ax * NUDGE;

    Minimise min{f, f, init.size()};

    if (!min.findMin(old) || !min.findMin(end)) {
        // failed minimisation
        return {2, Vector{}, Vector{}};
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
        // std::cout << distOld << std::endl;
        // std::cout << distFwd << std::endl;
        return {3, Vector{}, Vector{}};
    }

    if (dot(end - old, end - old) < TOL_NEAR) {
        // minimasations both converged to init
        return {4, Vector{}, Vector{}};
    }

    // output(end, col); // tmp

    return {0, std::move(sp), std::move(end)};
}

inline constexpr double ARRHENIUS_PRE = 1e13;
inline constexpr double KB_T = 8.617333262145 * 1e-5 * 500; // eV K^-1

inline constexpr double INV_KB_T = 1 / KB_T;

double activeToRate(double active_E) {

    check(active_E > 0, "sp energy < init energy");

    return ARRHENIUS_PRE * std::exp(-active_E * INV_KB_T);
}

//
template <typename T, typename K>
void updateCatalog(std::unordered_map<Rdf, Topology> &catalog, Vector const &x,
                   T const &f, TopoClassify<K> const &cl) {

    double f_x = f(x);

    for (std::size_t i = 0; i < cl.size(); ++i) {

        catalog[cl.getRdf(i)].count += 1; // default constructs new

        std::size_t count = catalog[cl.getRdf(i)].count;

        while (catalog[cl.getRdf(i)].sp_searches < 25 ||
               catalog[cl.getRdf(i)].sp_searches < std::sqrt(count)) {

            ++(catalog[cl.getRdf(i)].sp_searches);

            std::cout << "@ " << i << '/' << cl.size() << " dimer launch ... "
                      << std::flush;

            auto [err, sp, end] = findSaddle(x, i, f);

            if (!err) {
                std::cout << "success!" << std::endl;

                auto [centre, ref] = cl.classifyMech(x, end);

                catalog[cl.getRdf(centre)].pushMech(f(sp) - f_x, f(end) - f_x,
                                                    std::move(ref));

            } else {
                std::cout << "err:" << err << std::endl;
            }
        }
    }
}

template <typename T, typename K>
void outputAllMechs(std::unordered_map<Rdf, Topology> &catalog, Vector const &x,
                    TopoClassify<K> const &cl, T const &f) {

    std::unordered_set<Rdf> done{};

    std::cout << "writing all mechanisms" << std::endl;

    output(x);

    for (std::size_t i = 0; i < cl.size(); ++i) {
        if (done.count(cl.getRdf(i)) == 0) {
            done.insert(cl.getRdf(i));

            for (auto &&m : catalog[cl.getRdf(i)].getMechs()) {
                std::cout << FRAME << ' ' << m.active_E << ' ' << m.delta_E
                          << std::endl;
                Vector recon = cl.reconstruct(x, i, m.ref);

                output(recon, f.quasiColourAll(recon));
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
    double delta_E;
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

    TabEAM data = parseTabEAM("/home/cdt1902/dis/CATkS/data/PotentialA.fs");

    Box box{
        data.rCut, 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    TopoClassify classifyer{box, kinds};

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

    std::unordered_map<Rdf, Topology> catalog = readMap("dump");

    double time = 0;

    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::uniform_real_distribution<> uniform_dist(0, 1);

    // //////////////////OLD Reconstruct test////////////////////////////
    //
    // classifyer.loadAtoms(init);
    //
    // classifyer.verify(init);
    //
    // std::cout << "loaded" << std::endl;
    //
    // output(init);
    //
    // while (true) {
    //     auto [err, sp, end] = findSaddle(init, 12, f);
    //
    //     if (!err) {
    //         output(end);
    //
    //         auto [centre, ref] = classifyer.classifyMech(init, end);
    //
    //         std::cout << "\nCentre was " << centre << std::endl;
    //
    //         Vector recon = classifyer.reconstruct(init, 12, ref);
    //
    //         std::cout << f(end) - f(init) << std::endl;
    //         std::cout << f(recon) - f(init) << std::endl;
    //
    //         output(recon);const T &f
    //
    //         min.findMin(recon);
    //
    //         output(recon);
    //
    //         return 0;
    //     }
    // }
    //
    // ////////////////////////////////////////////////////

    for (int anon = 0; anon < 5; ++anon) {

        // if (anon == 0) {
        //     outputAllMechs(catalog, init, f);
        //     return 0;
        // }

        output(init, f.quasiColourAll(init));

        classifyer.loadAtoms(init);

        classifyer.verify(init);

        std::cout << "All topos verified!" << std::endl;

        updateCatalog(catalog, init, f, classifyer);

        writeMap("dump", catalog);

        //    outputAllMechs(catalog, init, classifyer, f);

        std::vector<LocalisedMech> possible{};

        for (std::size_t i = 0; i < classifyer.size(); ++i) {
            for (auto &&m : catalog[classifyer.getRdf(i)].getMechs()) {
                possible.push_back(
                    {i, activeToRate(m.active_E), m.ref, m.delta_E});
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

        Vector next = classifyer.reconstruct(init, choice.atom, choice.ref);

        std::cout << "here" << std::endl;

        double delta = f(next) - f(init);

        std::cout << "Goal: " << choice.delta_E << " actual: " << delta
                  << std::endl;

        check(std::abs(delta - choice.delta_E) < 0.1, "recon err");

        init = next;

        min.findMin(init);

        std::cout << "TIME: " << time << std::endl;

        // f.sort(init);
    }
}
