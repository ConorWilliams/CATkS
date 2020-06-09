// #define NDEBUG
// #define EIGEN_NO_DEBUG

#include <iostream>
#include <random>
#include <tuple>

#include "pcg_random.hpp"

#include "Dimer.hpp"
#include "DumpXYX.hpp"
#include "Forces.hpp"
#include "Minimise.hpp"
#include "utils.hpp"

// controls displacement along mm at saddle
inline constexpr double NUDGE = 0.1;
// tollerence for 3N vectors to be considered the same vector
inline constexpr double TOL_NEAR = 0.1;

constexpr int IDX = 9;
constexpr double G_SPHERE = 3;
constexpr double G_AMP = 0.325;

int FRAME = 0;
static const std::string head{"/home/cdt1902/dis/CATkS/plt/dump/all_"};
static const std::string tail{".xyz"};

template <typename T = std::vector<int>>
void output(Vector const &x, T const &kinds) {
    dumpXYX(head + std::to_string(FRAME++) + tail, x, kinds);
}

void output(Vector const &x) { output(x, std::vector<int>(x.size() / 3, 0)); }

// init is a minimised (unporturbed) vector of atoms
// f is their force object
template <typename T>
std::tuple<int, Vector, Vector> findSaddle(Vector const &init, T const &f) {
    Vector sp = init;
    Vector ax = Vector::Zero(init.size());

    // random pertabations

    // Seed with a real random value, if available
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    std::normal_distribution<> gauss_dist(0, G_AMP);

    double x = init[IDX * 3 + 0];
    double y = init[IDX * 3 + 1];
    double z = init[IDX * 3 + 2];

    std::vector<int> col(init.size() / 3, 0);

    for (int i = 0; i < init.size(); i = i + 3) {
        double dist = f.norm(x, y, z, init[i + 0], init[i + 1], init[i + 2]);

        if (dist < G_SPHERE) {
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

    output(sp, col); // tmp

    Dimer dimer{f, sp, ax, [&]() { output(sp, col); }};

    if (!dimer.findSaddle()) {
        // failed SP search
        return std::make_tuple(1, Vector{}, Vector{});
    }

    Vector old = sp + ax * NUDGE;
    Vector end = sp - ax * NUDGE;

    Minimise min{f, f, init.size()};

    if (!min.findMin(old) || !min.findMin(end)) {
        // failed minimisation
        return std::make_tuple(2, Vector{}, Vector{});
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
        std::cout << distOld << std::endl;
        std::cout << distFwd << std::endl;
        return std::make_tuple(3, Vector{}, Vector{});
    }

    if (dot(end - old, end - old) < TOL_NEAR) {
        // minimasations both converged to init
        return std::make_tuple(4, Vector{}, Vector{});
    }

    output(end, col); // tmp

    return std::make_tuple(0, std::move(sp), std::move(end));
}

struct Topo {
    double x;
    double y;
    double z;
    double r_sum;
};

template <typename T>
auto classifyMech(Vector const &init, Vector const &end, T const &f) {
    std::size_t centre = 0;

    { // find furthest moved
        double dr_max = 0;

        for (int i = 0; i < end.size(); i += 3) {
            double dr = f.norm(end[i + 0], end[i + 1], end[i + 2], init[i + 0],
                               init[i + 1], init[i + 2]);

            if (dr > dr_max) {
                centre = i / 3;
                dr_max = dr;
            }
        }
    }

    std::vector<int> col(init.size() / 3, 0);
    col[centre] = 1;
    output(end, col);

    // build list of all atoms within rcut of centre
    std::vector<std::size_t> near_atoms;

    double cx = init[3 * centre + 0];
    double cy = init[3 * centre + 1];
    double cz = init[3 * centre + 2];

    for (int i = 0; i < init.size(); i = i + 3) {
        double dist = f.norm(cx, cy, cz, init[i + 0], init[i + 1], init[i + 2]);

        if (dist < 4) {
            near_atoms.push_back(i);
            col[i / 3] = 1;
        }
    }

    output(init, col);

    ////////

    std::vector<Rdf> near_rdfs;

    for (auto &&a : near_atoms) {
        for (auto &&n : near_atoms) {
            if (&a != &n) {
                a.r_sum += f.norm(a.x, a.y, a.z, n.x, n.y, n.z);
            }
        }
    }

    std::sort(near_atoms.begin(), near_atoms.end(),
              [](auto l, auto r) { return l.r_sum < r.r_sum; });

    for (auto &&elem : near_atoms) {
        std::cout << elem.r_sum << std::endl;
    }

    //

    std::cout << "working " << col.size() << ' ' << init.size() << std::endl;
}

enum : uint8_t { Fe = 0, H = 1 };
constexpr double LAT = 2.855700;

inline constexpr int len = 5;

static const std::string OUTFILE = "/home/cdt1902/dis/CATkS/raw.txt";

int main() {
    Vector init(len * len * len * 3 * 2 - 3);
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

    // init[init.size() - 6] = LAT * 0.5;
    // init[init.size() - 5] = LAT * 0.25;
    // init[init.size() - 4] = LAT * 0;

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

    Minimise min{f, f, init.size()};

    min.findMin(init);

    for (int i = 0; i < 50; ++i) {
        std::cout << "this is cycle " << i << std::endl;

        auto [err, sp, end] = findSaddle(init, f);

        if (err != 0) {
            std::cerr << err << std::endl;
            return 0;
        }

        classifyMech(init, end, f);

        return 0;
    }

    return 0;
}

/* Flow
 * compute all topology keys
 * for each topo key not in DB do SP searches
 * for each relevent topology reconstruct transitions, add to db
 */
