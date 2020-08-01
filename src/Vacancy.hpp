#pragma once

#include <array>
#include <iomanip>

#include "Canon.hpp"
#include "Cell.hpp"
#include "DumpXYX.hpp"

template <std::size_t N>
class FindVacancy : public CellListSorted<AtomSortBase> {
  private:
    struct Kmean {
        Eigen::Vector3d pos = {2.855, 2.855, 2.855};
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        std::size_t size = 0;
    };

    std::array<Kmean, N> means{};
    std::vector<Eigen::Vector3d> under{};

    double sq_dist(Eigen::Vector3d const &dr) {
        return box.minImage(dr).squaredNorm();
    };

    void refineKmeans() {
        for (Eigen::Vector3d const &atom : under) {
            auto min_iter = std::min_element(
                means.begin(), means.end(),
                [&](Kmean const &a, Kmean const &b) -> bool {
                    return sq_dist(a.pos - atom) < sq_dist(b.pos - atom);
                });
            // std::cout << min_iter->pos.transpose() << '\n';
            min_iter->sum += box.minImage(atom - min_iter->pos);
            min_iter->size += 1;
        }

        for (Kmean &mean : means) {
            if (mean.size > 0) {
                mean.pos += mean.sum / mean.size;

                auto [x, y, z] =
                    box.mapIntoCell(mean.pos[0], mean.pos[1], mean.pos[2]);

                mean.pos = Eigen::Vector3d{x, y, z};
                mean.sum = Eigen::Vector3d::Zero();
                mean.size = 0;
            }

            // std::cout << "mean " << mean.pos.transpose() << '\n';
        }
    }

  public:
    using CellListSorted::CellListSorted;

    void find(Vector const &x) {
        fill(x);
        under.clear();

        for (auto &&atom : *this) {
            std::size_t order = 0;
            if (atom.kind() == 0) {
                forEachNeigh(atom, [&](auto const &neigh, double r, double,
                                       double, double) {
                    if (neigh.kind() == 0 && r < F_F_BOND) {
                        ++order;
                    }
                });

                if (order != 8) {
                    under.push_back(atom.pos());
                }
            }
        }

        // std::cout << "SIZE " << under.size() << '\n';

        for (std::size_t i = 0; i < (N == 1 ? 1 : 2); ++i) {
            refineKmeans();
        }
    }

    void output(Vector const &x, std::vector<int> const &kinds) {
        find(x);
        std::vector<int> col = kinds;
        Vector xs{x.size() + 3 * means.size()};

        xs.block(0, 0, x.size(), 1) = x;

        for (std::size_t i = 0; i < means.size(); ++i) {
            xs.block(x.size() + 3 * i, 0, 3, 1) = means[i].pos;
            col.push_back(1);
        }

        ::output(xs, col);
    }

    // void dump(std::string const &file, double time, Vector const &x) {
    //
    //     find(x);
    //
    //     std::ofstream outfile{file, std::ios::app};
    //
    //     outfile << time;
    //
    //     for (auto &&m : means) {
    //         outfile << ' ' << m.pos[0];
    //         outfile << ' ' << m.pos[1];
    //         outfile << ' ' << m.pos[2];
    //     }
    //
    //     outfile << "\n";
    // }

    void dump(std::string const &file, double time, double delta_E,
              Vector const &x, std::vector<int> const &kinds) {

        find(x);

        std::ofstream outfile{file, std::ios::app};

        outfile << std::fixed << std::setprecision(15);

        outfile << time << ' ' << delta_E;

        for (auto &&m : means) {
            outfile << ' ' << m.pos[0];
            outfile << ' ' << m.pos[1];
            outfile << ' ' << m.pos[2];
        }

        for (std::size_t i = 0; i < kinds.size(); ++i) {
            if (kinds[i] == 1) {
                for (std::size_t j = 0; j < 3; ++j) {
                    outfile << ' ' << x[3 * i + j];
                }
            }
        }

        outfile << "\n";
    }
};
