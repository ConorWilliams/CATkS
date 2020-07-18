#define NCHECK

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
#include "Catalog.hpp"
#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Forces2.hpp"
#include "Sort.hpp"
#include "Topo.hpp"
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

struct LocalisedMech {
    std::size_t atom;
    double rate;
    std::vector<Eigen::Vector3d> const &ref;
    double delta_E;
    double active_E;
};

double activeToRate(double active_E) {

    CHECK(active_E > 0, "sp energy < init energy " << active_E);

    return ARRHENIUS_PRE * std::exp(active_E * -INV_KB_T);
}

// template <typename T, std::size_t Rank> class Drray {
//   private:
//     T *m_data;
//     std::size_t m_stride[Rank];
//
//   public:
//     template <typename... Args> Drray(Args... dims) : m_stride{dims...} {
//         for (std::size_t i = 1; i < Rank; i++) {
//             m_stride[i] += m_stride[i - 1];
//         }
//     }
// };

int main(int argc, char **argv) {

    // CHECK(false, "false");

    VERIFY(argc == 3, "need an EAM data file and H dump file");

    Vector init(len * len * len * 3 * 2 + 3 * 3);
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if (false/*(i == 1 && j == 1 && k == 1) ||
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
    init[init.size() - 3] = LAT * (1 + 0.50);
    init[init.size() - 2] = LAT * (1 + 0.25);
    init[init.size() - 1] = LAT * (1 + 1.00);

    kinds[init.size() / 3 - 2] = H;
    init[init.size() - 6] = LAT * (1 + 0.25);
    init[init.size() - 5] = LAT * (2 + 0.00);
    init[init.size() - 4] = LAT * (1 + 0.50);

    kinds[init.size() / 3 - 3] = H;
    init[init.size() - 9] = LAT * (4 + 0.50);
    init[init.size() - 8] = LAT * (1 + 0.25);
    init[init.size() - 7] = LAT * (4 + 1.00);

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

    TopoClassify<Canon_t> classifyer{topo_box, kinds};

    Force_t f{force_box, kinds, data};

    // FindVacancy<2> v{force_box, kinds};
    // for (int _ = 0; _ < 3; ++_) {
    //     v.find(init);
    // }

    Catalog<Canon_t> catalog;

    Minimise min{f, f, init.size()};

    std::cout << "Before min" << std::endl;

    min.findMin(init);

    std::cout << "After min" << std::endl;

    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::uniform_real_distribution<> uniform_dist(0, 1);

    std::vector<LocalisedMech> possible{};

    while (iter < 10'000'000) {

        // v.output(init, f.quasiColourAll(init));
        // v.dump(argv[2], time, init);
        // output(init, f.quasiColourAll(init));
        dumpH(argv[2], time, init, kinds);

        ////////////////////////////////////////////////////////////

        classifyer.analyzeTopology(init);

        std::vector<size_t> idxs = catalog.getSearchIdxs(classifyer);

        int new_topos = classifyer.verify();

        int new_mechs = catalog.update(
            init, f, classifyer,
            [&](Eigen::Vector3d dr) { return topo_box.minImage(dr); }, idxs);

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
                // if (iter % 100 == 0 && m.active_E < 0.1) {
                //    std::cout << "bias" << std::endl;
                //} else {
                possible.push_back({
                    i,
                    activeToRate(m.active_E),
                    m.ref,
                    m.delta_E,
                    m.active_E,
                });
                //}
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
            std::cout << "hit end of choice" << std::endl;
            std::terminate();
        }();

        ////////////////////////////////////////////////////

        time += -std::log(uniform_dist(rng)) / rate_sum;

        const double energy_pre = f(init);

        init = classifyer.reconstruct(choice.atom, choice.ref);

        const double energy_recon = f(init) - energy_pre;

        min.findMin(init);

        const double energy_final = f(init) - energy_pre;

        const double rate = choice.rate;

        std::cout << "Memory:  " << choice.delta_E << '\n';
        std::cout << "Recon:   " << energy_recon << '\n';
        std::cout << "Final:   " << energy_final << '\n';
        std::cout << "Barrier: " << choice.active_E << "\n";
        std::cout << "Rate:    " << rate << " : " << rate / rate_sum << '\n';

        VERIFY(std::abs(energy_recon - choice.delta_E) < 0.1, "recon err");
        VERIFY(std::abs(energy_final - choice.delta_E) < 0.1, "recon err");

        std::cout << iter++ << " TIME: " << time << "\n\n";

        if (iter % 10000 == 0) {
            classifyer.write();
            catalog.write();
        }
    }

    classifyer.write();
    catalog.write();
}
