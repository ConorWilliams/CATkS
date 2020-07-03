// #define NDEBUG
// #define EIGEN_NO_DEBUG

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

double activeToRate(double active_E) {

    CHECK(active_E > 0, "sp energy < init energy");

    return ARRHENIUS_PRE * std::exp(-active_E * INV_KB_T);
}

// template <typename T, typename K>
// void outputAllMechs(
//     std::unordered_map<typename NautyCanon::Key_t, Topology> &catalog,
//     Vector const &x, K const &cl, T const &f) {
//
//     std::unordered_set<typename NautyCanon::Key_t> done{};
//
//     std::cout << "writing all mechanisms" << std::endl;
//
//     output(x);
//
//     for (std::size_t i = 0; i < cl.size(); ++i) {
//         if (done.count(cl.getRdf(i)) == 0) {
//             done.insert(cl.getRdf(i));
//
//             for (auto &&m : catalog[cl.getRdf(i)].getMechs()) {
//                 std::cout << FRAME << ' ' << m.active_E << ' ' << m.delta_E
//                           << std::endl;
//                 Vector recon = cl.reconstruct(i, m.ref);
//
//                 output(recon, f.quasiColourAll(recon));
//             }
//         }
//     }
// }

enum : uint8_t { Fe = 0, H = 1 };
constexpr double LAT = 2.855700;

inline constexpr int len = 5;

static const std::string OUTFILE = "/home/cdt1902/dis/CATkS/raw.txt";

struct LocalisedMech {
    std::size_t atom;
    double rate;
    std::vector<Eigen::Vector3d> const &ref;
    double delta_E;
    double active_E;
};

template <typename Atom_t> void play(std::vector<Atom_t> const &atoms) {
    Eigen::Matrix3d tensor = Eigen::Matrix3d::Zero();

    for (Atom_t const &a : atoms) {
        for (Atom_t const &b : atoms) {
            Eigen::Vector3d rab = a - b;

            tensor.noalias() += Eigen::Matrix3d::Identity() * rab.squaredNorm();
            tensor.noalias() -= rab * rab.transpose();
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(tensor);

    std::cout << solver.eigenvalues().transpose() << std::endl;

    std::cout << solver.eigenvectors() << std::endl;
}

int main() {

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

    using Canon_t = NautyCanon;

    TabEAM const data =
        parseTabEAM("/home/cdt1902/dis/CATkS/data/PotentialA.fs");

    Box const force_box{
        data.rCut, 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    Box const topo_box{
        data.rCut, 0, len * LAT, 0, len * LAT, 0, len * LAT,
    };

    TopoClassify<Canon_t> classifyer{topo_box, kinds};

    FuncEAM f{force_box, kinds, data};

    Catalog<Canon_t> catalog;

    // f.sort(init);

    Minimise min{f, f, init.size()};

    std::cout << "before" << std::endl;

    min.findMin(init);

    std::cout << "after" << std::endl;

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

        dumpH("h_diffusion.xyz", init, kinds);

        classifyer.analyzeTopology(init);

        // classifyer.colourPrint();

        classifyer.verify();

        catalog.update(init, f, classifyer, [&](Eigen::Vector3d dr) {
            return topo_box.minImage(dr);
        });

        classifyer.write();
        catalog.write();

        //    outputAllMechs(catalog, init, classifyer, f);

        // outputAllMechs(catalog, init, f);

        // if (anon == 0) {
        //     return 0;
        // }

        std::vector<LocalisedMech> possible{};

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

        time += -std::log(uniform_dist(rng)) / rate_sum;

        double f_x = f(init);

        init = classifyer.reconstruct(choice.atom, choice.ref);

        // output(init, f.quasiColourAll(init));

        double recon = f(init) - f_x;

        min.findMin(init);

        // output(init, f.quasiColourAll(init));

        double relax = f(init) - f_x;

        std::cout << "Memory:  " << choice.delta_E << '\n';
        std::cout << "Recon:   " << recon << '\n';
        std::cout << "Relaxed: " << relax << '\n';
        std::cout << "Barrier: " << choice.active_E << "\n";
        std::cout << "Rate:    " << activeToRate(choice.active_E) << ' '
                  << activeToRate(choice.active_E) / rate_sum << '\n';

        CHECK(std::abs(recon - choice.delta_E) < 0.1, "recon err");

        std::cout << "TIME: " << time << '\n' << std::endl;

        // f.sort(init);
    }
}

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
