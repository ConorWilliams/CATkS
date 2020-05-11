#include <random>

#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "FuncEAM.hpp"
#include "Minimise.hpp"
#include "utils.hpp"

enum : uint8_t { Fe = 0, H = 1 };
constexpr double LAT = 2.855700;

int main() {
    Vector x(7 * 7 * 7 * 3 * 2);
    Vector grad(x.size());

    std::vector<int> kinds(x.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            for (int k = 0; k < 7; ++k) {

                x[3 * cell + 0] = i * LAT;
                x[3 * cell + 1] = j * LAT;
                x[3 * cell + 2] = k * LAT;

                x[3 * cell + 3] = (i + 0.5) * LAT;
                x[3 * cell + 4] = (j + 0.5) * LAT;
                x[3 * cell + 5] = (k + 0.5) * LAT;

                cell += 2;
            }
        }
    }

    x[0] = LAT * 0.5;
    x[1] = LAT * (0.5 - 0.1);
    x[2] = 0;

    kinds[0] = H;

    FuncEAM f{"/home/cdt1902/dis/CATkS/data/PotentialA.fs",
              kinds,
              0,
              7 * LAT,
              0,
              7 * LAT,
              0,
              7 * LAT};

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // std::normal_distribution<double> d{0, 0.1};
    std::uniform_real_distribution<double> random(-1, 1);

    Vector ax(x.size());

    for (auto &&elem : ax) {
        elem = random(gen);
    }
    ax.matrix().normalize();

    dumpXYX("/home/cdt1902/dis/CATkS/plt/dump/test.pre.xyz", x, kinds);

    Dimer dimer{f, x, ax};

    dimer.findSaddle();

    // Minimise min{f, f, x.size()};

    // min.findMin(x);

    dumpXYX("/home/cdt1902/dis/CATkS/plt/dump/test.xyz", x, kinds);

    return 0;
}
