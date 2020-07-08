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
 * Make Cell have fill & rebuildGhosts only
 *
 */

#include <cmath>
#include <iostream>
#include <limits>

#include "DumpXYX.hpp"

#include "Canon.hpp"
#include "Catalog.hpp"
#include "Dimer.hpp"

#include "Forces.hpp"
#include "Topo.hpp"
#include "utils.hpp"

inline constexpr double ARRHENIUS_PRE = 5.12e12;
inline constexpr double KB_T = 8.617333262145 * 1e-5 * 300; // eV K^-1

inline constexpr double INV_KB_T = 1 / KB_T;

inline constexpr double ACCUMULATED_ERROR_LIMIT = 0.05; // eV

enum : uint8_t { Fe = 0, H = 1 };

constexpr double LAT = 2.855700;

inline constexpr int len = 15;

#include "Spline.hpp"

int main(int argc, char **argv) {

    // CHECK(false, "false");

    VERIFY(argc == 3, "need an EAM data file and H dump file");

    Vector init(len * len * len * 3 * 2 + 3 * -1);
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if ((i == 0 && j == 0 && k == 0)/* ||
                    (i == 4 && j == 1 && k == 1)*/) {
                    init[3 * cell + 0] = (i)*LAT;
                    init[3 * cell + 1] = (j)*LAT;
                    init[3 * cell + 2] = (k)*LAT;

                    std::cout << cell << std::endl;

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

    // kinds[init.size() / 3 - 1] = H;
    //
    // init[init.size() - 3] = LAT * (0 + 0.50);
    // init[init.size() - 2] = LAT * (0 + 0.25);
    // init[init.size() - 1] = LAT * (0 + 0.00);

    std::cout << "Loading " << argv[1] << '\n';

    static TabEAM const data = parseTabEAM(argv[1]);

    static Box const force_box{
        data.rcut(), 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    FuncEAM f{force_box, kinds, data};

    Minimise min{f, f, init.size()};

    std::cout << "before minim " << kinds.size() << std::endl;

    min.findMin(init);

    std::cout << "after min" << std::endl;

    std::cout << std::setprecision(15);

    std::cout << "total energy" << f(init) << std::endl;

    while (true) {
        std::vector<std::tuple<Vector, Vector>> search =
            findSaddle(1, init, 0, f,
                       [&](auto const &dr) { return force_box.minImage(dr); });

        if (search.size() > 0) {

            auto &&[sp, end] = search[0];

            std::cout << "init     :" << f(init) << '\n';
            std::cout << "sp       :" << f(sp) << '\n';
            std::cout << "end      :" << f(end) << '\n';

            dumpXYX("init.xyz", init, kinds);
            dumpXYX("sp.xyz", sp, kinds);
            dumpXYX("end.xyz", end, kinds);

            double barrier = f(sp) - f(init);
            double delta = f(end) - f(init);

            std::cout << "barrier  :" << barrier << "\n";
            std::cout << "delta    :" << delta << "\n";

            output(end, f.quasiColourAll(end));

            //////force////////////////////////

            return 0;
        }
    }
}
