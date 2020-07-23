//#define NCHECK

#define EIGEN_NO_DEBUG
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
 * Make Cell have fill & rebuildGhosts only.
 *
 */

#include <cmath>
#include <iostream>
#include <limits>

#include "Canon2.hpp"
#include "Catalog2.hpp"
#include "Cbuff.hpp"
#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Forces2.hpp"
#include "Sort.hpp"
#include "Vacancy.hpp"
#include "utils.hpp"

inline constexpr double ARRHENIUS_PRE = 5.12e12;
inline constexpr double TEMP = 300;                              // k
inline constexpr double KB_T = 1380649.0 / 16021766340.0 * TEMP; // eV K^-1

inline constexpr double INV_KB_T = 1 / KB_T;

enum : uint8_t { Fe = 0, H = 1 };

constexpr double LAT = 2.855700;

using Canon_t = NautyCanon2;
using Force_t = FuncEAM2;

inline constexpr int len = 7;

double activeToRate(double active_E) {

    CHECK(active_E > 0, "sp energy < init energy " << active_E);

    return ARRHENIUS_PRE * std::exp(active_E * -INV_KB_T);
}

struct Transition {
    std::size_t atom_idx;
    std::size_t topo_hash;
    std::size_t mech_id;

    Mechanism *mech;
    double rate;

    Transition(std::size_t atom_idx, std::size_t topo_hash, std::size_t mech_id,
               Mechanism &mech)
        : atom_idx{atom_idx}, topo_hash{topo_hash},
          mech_id(mech_id), mech{&mech}, rate{activeToRate(mech.active_E)} {}

    inline bool operator==(Transition const &other) const {
        return atom_idx == other.atom_idx && topo_hash == other.topo_hash &&
               mech_id == other.mech_id;
    }
};

int main(int argc, char **argv) {

    // CHECK(false, "false");

    VERIFY(argc == 3, "need an EAM data file and H dump file");

    Vector init(len * len * len * 3 * 2 + 3 * 0);
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if ((i == 1 && j == 1 && k == 1) /*||
                    (i == 2 && j == 1 && k == 1) ||
                    (i == 2 && j == 2 && k == 2) */) {
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
    init[init.size() - 3] = LAT * (1 + 0.5);
    init[init.size() - 2] = LAT * (1);
    init[init.size() - 1] = LAT * (1);

    // kinds[init.size() / 3 - 2] = H;
    // init[init.size() - 6] = LAT * (1 + 0.25);
    // init[init.size() - 5] = LAT * (2 + 0.00);
    // init[init.size() - 4] = LAT * (1 + 0.50);
    //
    // kinds[init.size() / 3 - 3] = H;
    // init[init.size() - 9] = LAT * (4 + 0.50);
    // init[init.size() - 8] = LAT * (1 + 0.25);
    // init[init.size() - 7] = LAT * (4 + 1.00);

    // kinds[init.size() / 3 - 4] = H;
    // init[init.size() - 12] = LAT * (4 + 0.50);
    // init[init.size() - 11] = LAT * (4 + 0.25);
    // init[init.size() - 10] = LAT * (4 + 1.00);

    ////////////////////////////////////////////////////////////

    double time = 0;
    int iter = 1;

    std::cout << "Loading " << argv[1] << '\n';

    static const TabEAM data = parseTabEAM(argv[1]);

    static const Box force_box{
        data.rcut(), 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    static const Box topo_box = force_box;

    cellSort(init, kinds, force_box);

    Catalog<Canon_t> catalog{topo_box, kinds};

    Force_t f{force_box, kinds, data};

    FindVacancy<1> v{force_box, kinds};
    for (int _ = 0; _ < 3; ++_) {
        v.find(init);
    }

    Minimise min{f, f, init.size()};

    std::cout << "Before min" << std::endl;

    min.findMin(init);

    std::cout << "After min" << std::endl;

    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::uniform_real_distribution<> uniform_dist(0, 1);

    std::vector<Transition> possible{};

    Cbuff<Transition> kernal(5);

    v.dump(argv[2], 0, 0, init, kinds);

    while (iter < 10'000'000) {

        v.output(init, f.quasiColourAll(init));

        // output(init, f.quasiColourAll(init));
        // dumpH(argv[2], time, init, kinds);

        ////////////////////////////////////////////////////////////

        int new_mechs = catalog.update(
            init, f, [&](Eigen::Vector3d dr) { return topo_box.minImage(dr); });

        if (new_mechs > 0) {
            catalog.write();
        }

        //////////////////////////////////////////////////////////////

        possible.clear();

        for (std::size_t i = 0; i < catalog.size(); ++i) {
            auto [key, topo] = catalog[i];

            auto hash = std::hash<typename Canon_t::Key_t>{}(key);

            std::size_t mech_id = 0;

            for (auto &&mech : topo.mechs) {

                Transition tmp{i, hash, mech_id++, mech};

                if (!kernal.contains(tmp)) {
                    possible.emplace_back(std::move(tmp));
                }
            }
        }

        double rate_sum = std::accumulate(
            possible.begin(), possible.end(), 0.0,
            [](double sum, Transition const &m) { return sum + m.rate; });

        double p1 = uniform_dist(rng);

        Transition choice = [&]() {
            double sum = 0;
            for (auto &&elem : possible) {

                sum += elem.rate;
                if (sum > p1 * rate_sum) {
                    return elem;
                }
            }
            std::cout << "hit end of choice" << std::endl;
            std::terminate();
        }();

        ////////////////////////////////////////////////////

        time += -std::log(uniform_dist(rng)) / rate_sum;

        const double energy_pre = f(init);

        catalog.reconstruct(choice.atom_idx, init, *choice.mech);

        const double energy_recon = f(init) - energy_pre;

        min.findMin(init);

        const double energy_final = f(init) - energy_pre;

        const double rate = choice.rate;

        std::cout << "Memory:  " << choice.mech->delta_E << '\n';
        std::cout << "Recon:   " << energy_recon << '\n';
        std::cout << "Final:   " << energy_final << '\n';
        std::cout << "Barrier: " << choice.mech->active_E << "\n";
        std::cout << "Rate:    " << rate << " : " << rate / rate_sum << '\n';

        VERIFY(std::abs(energy_recon - choice.mech->delta_E) < 0.1,
               "recon err");
        VERIFY(std::abs(energy_final - choice.mech->delta_E) < 0.1,
               "recon err");

        std::cout << iter++ << " TIME: " << time << "\n\n";

        kernal.push_back(std::move(choice));

        if (iter % 10000 == 0) {
            catalog.write();
        }

        v.dump(argv[2], time, energy_final, init, kinds);
    }

    catalog.write();
}
