// #define NDEBUG
// #define EIGEN_NO_DEBUG

#include <iostream>
#include <limits>
#include <random>
#include <tuple>

#include "Eigen/Eigenvalues"
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
inline constexpr double NEG_TOL = -0.01;

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
        double dist =
            f.periodicNormSq(x, y, z, init[i + 0], init[i + 1], init[i + 2]);

        if (dist < G_SPHERE * G_SPHERE) {
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
    Eigen::Vector3d pos;
    int idx;
};

//
// template <typename It> It findRun(It beg, It end) {
//     check(beg != end, "cant have empty run!");
//
//     It run = beg;
//     while (run != end && run->isNear(*beg)) {
//         ++run;
//     }
//     return run;
// }
//
// template <typename T> void canonicalOrder(std::vector<Topo> &list, T
// const &f) {
//     for (auto &&a : list) {
//         std::for_each(list.begin(), list.end(), [&](Topo &n) {
//             a.sum += f.periodicNormSq(a.pos[0], a.pos[1], a.pos[2],
//             n.pos[0],
//                                       n.pos[1], n.pos[2]);
//         });
//     }
//
//     std::sort(list.begin(), list.end());
//
//     Eigen::Vector3d ghost = list.front().pos;
//
//     for (auto &&elem : list) {
//         std::cout << elem.sum << std::endl;
//     }
//
//     auto it = list.begin();
//
//     while (it != list.end()) {
//         auto run = findRun(it, list.end());
//
//         if (run - it == 1) {
//             ++it;
//             std::cout << "pass" << std::endl;
//         } else {
//             ghost = (it->pos + ghost) * 0.5;
//
//             std::for_each(it, list.end(), [&](Topo &atom) {
//                 atom.sum +=
//                     f.periodicNormSq(ghost[0], ghost[1], ghost[2],
//                     atom.pos[0],
//                                      atom.pos[1], atom.pos[2]);
//             });
//
//             std::sort(it, list.end());
//
//             std::cout << "************after***********" << std::endl;
//
//             for (auto &&elem : list) {
//                 std::cout << eleTOL_NEARm.sum << std::endl;
//             }
//         }
//     }
// }

bool near(double x, double y) { return std::abs(x - y) < 0.01; }

template <typename T>
Eigen::Vector3d findCentre(std::vector<Topo> const &list, T const &f) {
    // find center atom
    Eigen::Vector3d centre;
    double min = std::numeric_limits<double>::max();

    for (auto &&atom : list) {
        double sum = 0;
        for (auto &&other : list) {
            sum += f.minImage(atom.pos - other.pos).squaredNorm();
        }
        if (sum < min) {
            centre = atom.pos;
            min = sum;
        }
    }

    check(min != std::numeric_limits<double>::max(),
          "failed to find a centre?");

    return centre;
}

// Maps positions of atoms in list to relative positions in inerta basis.
// Sorts list into canonical order.
// Returns orthoganal transform matrix T sush that:
// T * x[inerta basis] = x[standard basis]
template <typename T>
Eigen::Matrix3d toInertaBasis(std::vector<Topo> &list, T const &f) {
    //
    check(list.size() > 1, "not enough atoms");
    std::cout << std::endl
              << "There are " << list.size() << " atoms" << std::endl;

    // find center atom
    Eigen::Vector3d centre = findCentre(list, f);

    // make inerta tensor (rel to centre) and map : pos -> rel_pos
    Eigen::Matrix3d I = Eigen::Matrix3d::Zero();

    for (auto &&atom : list) {
        atom.pos = f.minImage(atom.pos - centre);

        I.noalias() += atom.pos.squaredNorm() * Eigen::Matrix3d::Identity() -
                       atom.pos * atom.pos.transpose();
    }

    std::cout << std::endl << "Inerta tensor:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << I(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    // find eigen vectors and values
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(I);

    Eigen::Matrix3d eVecs = solver.eigenvectors();
    Eigen::Vector3d eVals = solver.eigenvalues();

    std::cout << std::endl << "Eigen values:" << std::endl;
    std::cout << eVals[0] << ' ' << eVals[1] << ' ' << eVals[2] << std::endl;

    std::cout << std::endl << "Eigen vectors:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << eVecs(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    // if symetric top move unique to front for better sorting
    if (near(eVals[0], eVals[1])) {
        eVecs.col(0).swap(eVecs.col(2));
        std::swap(eVals[0], eVals[2]);
    }

    // calculate projection sums
    Eigen::Vector3d sums = Eigen::Vector3d::Zero();
    for (auto &&atom : list) {
        sums.noalias() += eVecs.transpose() * atom.pos;
    }

    std::cout << std::endl << "Projection sums:" << std::endl;
    std::cout << sums[0] << ' ' << sums[1] << ' ' << sums[2] << std::endl;

    // flip axis as appropriate
    if (sums[0] < NEG_TOL) {
        eVecs.col(0) *= -1;
        std::cout << "flip 0" << std::endl;
    }
    if (sums[1] < NEG_TOL) {
        eVecs.col(1) *= -1;
        std::cout << "flip 1" << std::endl;
    }
    if (sums[2] < NEG_TOL) {
        eVecs.col(2) *= -1;
        std::cout << "flip 2" << std::endl;
    }

    // correct for symetric tops by fixing handedness of basis
    std::cout << std::endl << "handedness" << std::endl;
    double hand = eVecs.col(0).dot(eVecs.col(1).cross(eVecs.col(2)));

    std::cout << hand << std::endl;

    if (near(eVals[1], eVals[2]) && hand < 0) {
        eVecs.col(1).swap(eVecs.col(2));
        std::swap(eVals[1], eVals[2]);
    }

    /// map : rel_pos -> [corrected]_eigen_basis rel
    for (auto &&atom : list) {
        atom.pos = eVecs.transpose() * atom.pos;
    }

    // lex sort
    std::sort(list.begin(), list.end(), [&](Topo a, Topo b) {
        if (near(a.pos[0], b.pos[0])) {
            if (near(a.pos[1], b.pos[1])) {
                if (near(a.pos[2], b.pos[2])) {
                    return false;
                } else {
                    return a.pos[2] < b.pos[2];
                }
            } else {
                return a.pos[1] < b.pos[1];
            }
        } else {
            return a.pos[0] < b.pos[0];
        }
    });

    std::cout << std::endl << "new coords" << std::endl;

    for (auto &&atom : list) {
        std::cout << atom.pos[0] << ' ' << atom.pos[1] << ' ' << atom.pos[2]
                  << std::endl;
    }

    return eVecs;
}

template <typename T>
auto classifyMech(Vector const &init, Vector const &end, T const &f) {
    std::size_t centre = 0;

    { // find furthest moved
        double dr_sq_max = 0;

        for (int i = 0; i < end.size(); i += 3) {
            double dr_sq =
                f.periodicNormSq(end[i + 0], end[i + 1], end[i + 2],
                                 init[i + 0], init[i + 1], init[i + 2]);

            if (dr_sq > dr_sq_max) {
                centre = i / 3;
                dr_sq_max = dr_sq;
            }
        }
    }

    std::vector<int> col(init.size() / 3, 0);
    col[centre] = 1;
    output(end, col);

    // build list of all atoms within rcut of centre
    std::vector<Topo> near_atoms;

    double cx = init[3 * centre + 0];
    double cy = init[3 * centre + 1];
    double cz = init[3 * centre + 2];

    for (int i = 0; i < init.size(); i = i + 3) {
        double dist_sq =
            f.periodicNormSq(cx, cy, cz, init[i + 0], init[i + 1], init[i + 2]);

        if (dist_sq < 2.6 * 2.6) {
            near_atoms.push_back(
                {{init[i + 0], init[i + 1], init[i + 2]}, i / 3});
        }
    }

    for (std::size_t i = 0; i < near_atoms.size(); ++i) {
        col[near_atoms[i].idx] = 1;
    }

    output(init, col);

    //////// find canonical-ordering ///////////

    Eigen::Matrix3d transform = toInertaBasis(near_atoms, f);

    for (std::size_t i = 0; i < near_atoms.size(); ++i) {
        col[near_atoms[i].idx] = i + 1;
    }

    output(init, col);

    ////// store normalised /////
    for (auto &&atom : near_atoms) {
        atom.pos[0] = end[3 * atom.idx + 0] - init[3 * atom.idx + 0];
        atom.pos[1] = end[3 * atom.idx + 1] - init[3 * atom.idx + 1];
        atom.pos[2] = end[3 * atom.idx + 2] - init[3 * atom.idx + 2];

        atom.pos = transform.transpose() * f.minImage(atom.pos);
    }

    std::cout << std::endl << "memory" << std::endl;
    for (auto &&atom : near_atoms) {
        std::cout << atom.pos[0] << ' ' << atom.pos[1] << ' ' << atom.pos[2]
                  << std::endl;
    }

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

        while (true) {
            auto [err, sp, end] = findSaddle(init, f);

            if (err) {
                continue;
            }

            classifyMech(init, end, f);

            return 0;
        }
    }

    return 0;
}

/* Flow
 * compute all topology keys
 * for each topo key not in DB do SP searches
 * for each relevent topology reconstruct transitions, add to db
 */
