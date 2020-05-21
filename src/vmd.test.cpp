#define NDEBUG
#define EIGEN_NO_DEBUG

#include <iostream>
#include <random>

#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Minimise.hpp"
#include "utils.hpp"

enum : uint8_t { Fe = 0, H = 1 };
constexpr double LAT = 2.855700;

inline constexpr int len = 7;

static const std::string OUTFILE = "/home/cdt1902/dis/CATkS/raw.txt";
constexpr double G_AMP = 0.1;

int main() {
    Vector init(len * len * len * 3 * 2 - 3);
    Vector grad(init.size());
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if (i == 1 && j == 1 && k == 1) {
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
    //
    // init[init.size() - 6] = LAT * 0.5;
    // init[init.size() - 5] = LAT * 0.25;
    // init[init.size() - 4] = LAT * 0;
    //
    // kinds[kinds.size() - 2] = H;

    FuncEAM f{"/home/cdt1902/dis/CATkS/data/PotentialA.fs",
              kinds,
              0,
              len * LAT,
              0,
              len * LAT,
              0,
              len * LAT};

    f.sort(init);

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0, G_AMP};
    std::uniform_real_distribution<double> random(-1, 1);

    Minimise min{f, f, init.size()};

    min.findMin(init);

    auto unkinds = kinds;

    dumpXYX("/home/cdt1902/dis/CATkS/plt/dump/vac_rand.xyz", init, kinds);

    Vector x = init;

    int frame = 0;

    std::vector<Vector> to_print;
    std::vector<std::string> names;

    auto printer = [&]() {
        std::string head{"/home/cdt1902/dis/CATkS/plt/dump/vac_vid_"};
        std::string head2{"/home/cdt1902/dis/CATkS/plt/dump/vac_cont_"};

        std::string tail{".xyz"};
        to_print.push_back(x);
        // dumpXYX(head2 + std::to_string(frame) + tail, x, kinds);
        names.push_back(head + std::to_string(frame++) + tail);
    };

    Dimer dimer{f, x, ax, printer};

    std::ofstream file{OUTFILE};

    for (int i = 0; i < 500; ++i) {
        std::cout << "this is cycle " << i << std::endl;

        x = init;

        for (auto &&elem : ax) {
            elem = 0;
        }

        // was 35, vac 34
        for (std::size_t i = 0; i < 34 + 34; ++i) {
            x[3 * i + 0] += d(gen);
            x[3 * i + 1] += d(gen);
            x[3 * i + 2] += d(gen);

            ax[3 * i + 0] += random(gen);
            ax[3 * i + 1] += random(gen);
            ax[3 * i + 2] += random(gen);

            if (unkinds[i] == Fe) {
                unkinds[i] = 2;
            }
        }

        ax.matrix().normalize();

        // unkinds[8] = 3;
        // unkinds[9] = 3;

        bool ok = dimer.findSaddle();

        std::string head{"/home/cdt1902/dis/CATkS/plt/dump/vac_sad_"};
        std::string tail{"_.xyz"};

        std::string ver[2] = {"_fail", "_pass"};

        // dumpXYX(head + std::to_string(i) + ver[ok] + tail, x, unkinds);

        if (ok) {
            Vector plus = x + ax * 0.1;
            Vector minus = x - ax * 0.1;

            min.findMin(plus);
            min.findMin(minus);

            double d1 = dot(init - minus, init - minus);
            double dx = dot(init - x, init - x);
            double d2 = dot(init - plus, init - plus);

            if (d2 < d1) {
                std::swap(d1, d2);
                std::swap(plus, minus);
            }

            std::cout << "minus: " << d1 << ' ' << f(minus) << std::endl;
            std::cout << "saddle: " << dx << ' ' << f(x) << std::endl;
            std::cout << "plus: " << d2 << ' ' << f(plus) << std::endl;

            std::cout << "in    " << f(x) - f(minus) << std::endl;
            std::cout << "out   " << f(plus) - f(x) << std::endl;
            std::cout << "total " << f(plus) - f(minus) << std::endl;

            if (d1 < 0.1) {

                file << frame << ' ' << f(x) - f(minus) << ' ' << f(plus) - f(x)
                     << ' ' << f(plus) - f(minus) << std::endl;

                std::string head{"/home/cdt1902/dis/CATkS/plt/dump/vac_vid_"};
                std::string tail{".xyz"};
                to_print.push_back(plus);
                names.push_back(head + std::to_string(frame++) + tail);

                std::cout << "progressing << std::endl" << std::endl;
                init = plus;

                for (std::size_t i = 0; i < to_print.size(); ++i) {
                    dumpXYX(names[i], to_print[i], kinds);
                }

                f.sort(init);
            }
        }

        names.clear();
        to_print.clear();

        frame += 100;
    }

    // Vector x_cpy = x;
    //
    // x += ax;
    // x_cpy -= ax;
    //
    // min.findMin(x);
    // dumpXYX("/home/cdt1902/dis/CATkS/plt/dump/m1.xyz", x, kinds);
    //
    // min.findMin(x_cpy);
    // dumpXYX("/home/cdt1902/dis/CATkS/plt/dump/m2.xyz", x_cpy, kinds);

    return 0;
}
