//#define NCHECK
//#define EIGEN_NO_DEBUG

#define EIGEN_DONT_PARALLELIZE

/*
 * Move subclasses from classifyer classes to namespaces.
 *
 * Base class for mechanism and toporef::topo.
 *
 * Make force functor default constructable in thread and take kinds as
 *  parameter to force call, use static const member to refere to shard EAM data
 *  state.
 *
 * Make a compile_time_constants.hpp file
 *
 * Make minimiser default constructable take kinds in function call, take f as
 *  template parameter.
 *
 * Make dimer take f as template param
 *
 * Make Cell have fill & rebuildGhosts only
 *
 */

#include <cmath>
#include <iostream>
#include <limits>

#include "Canon.hpp"
#include "Catalog.hpp"
#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Topo.hpp"
#include "utils.hpp"

inline constexpr double ARRHENIUS_PRE = 5.12e12;
inline constexpr double KB_T = 8.617333262145 * 1e-5 * 300; // eV K^-1

inline constexpr double INV_KB_T = 1 / KB_T;

inline constexpr double ACCUMULATED_ERROR_LIMIT = 0.05; // eV

enum : uint8_t { Fe = 0, H = 1 };

constexpr double LAT = 2.855700;

inline constexpr int len = 5;

struct LocalisedMech {
    std::size_t atom;
    double rate;
    std::vector<Eigen::Vector3d> const &ref;
    double delta_E;
    double active_E;
};

double activeToRate(double active_E) {

    CHECK(active_E > 0, "sp energy < init energy");

    return ARRHENIUS_PRE * std::exp(active_E * -INV_KB_T);
}

int main(int argc, char **argv) {

    // CHECK(false, "false");

    VERIFY(argc == 2, "need a EAM data file");

    Vector init(len * len * len * 3 * 2 + 3 * 1);
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if (false /*(i == 1 && j == 1 && k == 1) ||
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

    kinds[init.size() / 3 - 1] = H;

    init[init.size() - 3] = LAT * (1 + 0.50);
    init[init.size() - 2] = LAT * (1 + 0.25);
    init[init.size() - 1] = LAT * (1 + 0.00);

    // kinds[init.size() / 3 - 2] = H;
    //
    // init[init.size() - 6] = LAT * (2 + 0.50);
    // init[init.size() - 5] = LAT * (2 + 0.25);
    // init[init.size() - 4] = LAT * (2 + 0.00);

    ////////////////////////////////////////////////////////////

    using Canon_t = NautyCanon;

    double time = 0;
    double energy_error = 0;
    int iter = 0;

    static const TabEAM data = parseTabEAM(argv[1]);

    static const Box force_box{
        data.rCut, 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    static const Box topo_box{
        data.rCut, 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    TopoClassify<Canon_t> classifyer{topo_box, kinds};

    FuncEAM f{force_box, kinds, data};

    Catalog<Canon_t> catalog;

    Minimise min{f, f, init.size()};

    std::cout << "before min" << std::endl;

    min.findMin(init);

    std::cout << "after min" << std::endl;

    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::uniform_real_distribution<> uniform_dist(0, 1);

    std::vector<LocalisedMech> possible{};

    double energy_pre = f(init);

    while (iter < 500) {

        output(init, f.quasiColourAll(init));

        dumpH("h_diffusion.xyz", time, init, kinds);

        ////////////////////////////////////////////////////////////

        if (energy_error > ACCUMULATED_ERROR_LIMIT) {
            min.findMin(init);
            energy_pre = f(init);
            energy_error = 0;
            classifyer.analyzeTopology(init);
        } else {
            classifyer.analyzeTopology(init);

            if (catalog.requireSearch(classifyer)) {
                min.findMin(init);
                energy_pre = f(init);
                energy_error = 0;
                classifyer.analyzeTopology(init);
            }
        }

        int new_topos = classifyer.verify();

        int new_mechs =
            catalog.update(init, f, classifyer, [&](Eigen::Vector3d dr) {
                return topo_box.minImage(dr);
            });

        if (new_topos > 0) {
            classifyer.write();
        }
        if (new_mechs > 0) {
            catalog.write();
        }

        //////////////////////////////////////////////////////////////

        possible.clear();

        for (std::size_t i = 0; i < classifyer.size(); ++i) {
            for (auto &&m : catalog[classifyer[i]]) {
                possible.push_back({
                    i,
                    activeToRate(m.active_E),
                    m.ref,
                    m.delta_E,
                    m.active_E,
                });
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

        ////////////////////////////////////////////////////

        time += -std::log(uniform_dist(rng)) / rate_sum;

        init = classifyer.reconstruct(choice.atom, choice.ref);

        const double energy_post = f(init);

        const double recon_delta_E = energy_post - energy_pre;

        energy_error += std::abs(recon_delta_E - choice.delta_E);

        std::cout << "\nError:   " << energy_error << '\n';

        const double rate = choice.rate;

        std::cout << "Memory:  " << choice.delta_E << '\n';
        std::cout << "Recon:   " << recon_delta_E << '\n';
        std::cout << "Barrier: " << choice.active_E << "\n";
        std::cout << "Rate:    " << rate << " : " << rate / rate_sum << '\n';

        VERIFY(std::abs(recon_delta_E - choice.delta_E) < 0.1, "recon err");

        std::cout << iter++ << " TIME: " << time << "\n\n";

        if (iter % 1000 == 0) {
            classifyer.write();
            catalog.write();
        }

        energy_pre = energy_post;
    }

    classifyer.write();
    catalog.write();
}
