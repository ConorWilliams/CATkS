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

#include "Canon.hpp"
#include "Catalog.hpp"
#include "Cbuff.hpp"
#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Sort.hpp"
#include "Superbasin.hpp"
#include "Vacancy.hpp"
#include "utils.hpp"

enum : uint8_t { Fe = 0, H = 1 };

constexpr double LAT = 2.855700;

using Canon_t = NautyCanon2;
using Force_t = FuncEAM2;

inline constexpr int len = 7;

int main(int argc, char **argv) {

    // CHECK(false, "false");

    VERIFY(argc == 3, "need an EAM data file and H dump file");

    Vector init(len * len * len * 3 * 2 + 3 * (1 - 2));
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if ((i == 1 && j == 1 && k == 1) ||
                    (i == 2 && j == 1 && k == 1)/* ||
                    (i == 3 && j == 1 && k == 1) */) {
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
    // init[init.size() - 6] = LAT * (0 + 0.50);
    // init[init.size() - 5] = LAT * (1 + 0.25);
    // init[init.size() - 4] = LAT * (1 + 0.00);

    // kinds[init.size() / 3 - 3] = H;
    // init[init.size() - 9] = LAT * (4 + 0.50);
    // init[init.size() - 8] = LAT * (1 + 0.25);
    // init[init.size() - 7] = LAT * (4 + 1.00);

    // kinds[init.size() / 3 - 4] = H;
    // init[init.size() - 12] = LAT * (4 + 0.50);
    // init[init.size() - 11] = LAT * (4 + 0.25);
    // init[init.size() - 10] = LAT * (4 + 1.00);

    init += LAT * 0.25 * std::sqrt(3); // atoms not at edge

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

    FindVacancy<2> v{force_box, kinds};

    for (int _ = 0; _ < 3; ++_) {
        v.find(init);
    }

    Minimise min{f, f, init.size()};

    std::cout << "Before min" << std::endl;

    min.findMin(init);

    std::cout << "After min" << std::endl;

    KineticMC sb;

    // v.dump(argv[2], 0, 0, init, kinds);

    while (iter < 10'000'000) {
        force_box.remap(init);

        // v.output(init, f.quasiColourAll(init));
        // output(init, f.quasiColourAll(init));
        // dumpH(argv[2], time, init, kinds);

        ////////////////////////////////////////////////////////////

        int new_mechs = catalog.update(
            init, f, [&](Eigen::Vector3d dr) { return topo_box.minImage(dr); });

        if (new_mechs > 0) {
            catalog.write();
        }

        // init may be changed during superbasin dynamics
        auto &&[dt, atom, mech] = sb.chooseMechanism(catalog, init);

        const double energy_pre = f(init);

        catalog.reconstruct(atom, init, mech);

        time += dt;

        const double energy_recon = f(init) - energy_pre;

        min.findMin(init);

        const double energy_final = f(init) - energy_pre;

        std::cout << "Forward: " << mech.active_E << '\n';
        std::cout << "Reverse: " << mech.active_E - mech.delta_E << '\n';

        std::cout << "Memory:  " << mech.delta_E << '\n';
        std::cout << "Relaxed: " << energy_final << '\n';
        std::cout << "Recon:   " << energy_recon << '\n';

        double diff = std::abs(energy_final - mech.delta_E);
        double frac = std::abs(diff / mech.delta_E);

        VERIFY(frac < 0.20 || diff < 0.037, "Reconstruction error!");

        std::cout << "ITER: " << iter++ << "; TIME: " << time << "\n\n";

        if (iter % 10000 == 0) {
            catalog.write();
        }

        v.dump(argv[2], time, mech.active_E, init, kinds);
    }

    catalog.write();
}
