#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "Classes.hpp"

#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Topo.hpp"
#include "utils.hpp"

inline constexpr double ARRHENIUS_PRE = 1e13;
inline constexpr double KB_T = 8.617333262145 * 1e-5 * 500; // eV K^-1

inline constexpr double INV_KB_T = 1 / KB_T;

double activeToRate(double active_E) {

    check(active_E > 0, "sp energy < init energy");

    return ARRHENIUS_PRE * std::exp(-active_E * INV_KB_T);
}

//
template <typename T, typename K>
void updateCatalog(std::unordered_map<Graph, Topology> &catalog,
                   Vector const &x, T const &f, TopoClassify<K> const &cl) {

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

                auto [centre, ref] = cl.classifyMech(end);

                catalog[cl.getRdf(centre)].pushMech(f(sp) - f_x, f(end) - f_x,
                                                    std::move(ref));

            } else {
                std::cout << "err:" << err << std::endl;
            }
        }
    }
}

template <typename T, typename K>
void outputAllMechs(std::unordered_map<Graph, Topology> &catalog,
                    Vector const &x, TopoClassify<K> const &cl, T const &f) {

    std::unordered_set<Graph> done{};

    std::cout << "writing all mechanisms" << std::endl;

    output(x);

    for (std::size_t i = 0; i < cl.size(); ++i) {
        if (done.count(cl.getRdf(i)) == 0) {
            done.insert(cl.getRdf(i));

            for (auto &&m : catalog[cl.getRdf(i)].getMechs()) {
                std::cout << FRAME << ' ' << m.active_E << ' ' << m.delta_E
                          << std::endl;
                Vector recon = cl.reconstruct(i, m.ref);

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

    Vector init(len * len * len * 3 * 2 - 6);
    Vector ax(init.size());

    std::vector<int> kinds(init.size() / 3, Fe);

    // make BCC lattice
    double cell = 0;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < len; ++j) {
            for (int k = 0; k < len; ++k) {

                if ((i == 1 && j == 1 && k == 1) ||
                    (i == 4 && j == 1 && k == 1)) {
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

    std::unordered_map<Graph, Topology> catalog = readMap("dump");

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

    for (int anon = 0; anon < 50; ++anon) {

        // if (anon == 0) {
        //     outputAllMechs(catalog, init, f);
        //     return 0;
        // }

        output(init, f.quasiColourAll(init));

        classifyer.analyzeTopology(init);

        classifyer.verify();

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

        double f_x = f(init);

        init = classifyer.reconstruct(choice.atom, choice.ref);

        std::cout << "here" << std::endl;

        double recon = f(init) - f_x;

        min.findMin(init);

        double relax = f(init) - f_x;

        std::cout << "Goal: " << choice.delta_E << " Recon: " << recon
                  << " Relaxed: " << relax << std::endl;

        check(std::abs(recon - choice.delta_E) < 0.1, "recon err");

        std::cout << "TIME: " << time << std::endl;

        // f.sort(init);
    }
}
