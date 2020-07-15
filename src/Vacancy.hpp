#pragma once

#include <array>

#include "Canon2.hpp"
#include "Cell2.hpp"
#include "DumpXYX.hpp"

template <std::size_t N>
class FindVacancy : public CellListSorted<AtomSortBase> {
  private:
    struct Kmean {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
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
        }
    }

  public:
    using CellListSorted::CellListSorted;

    std::vector<std::size_t> getVac(Vector const &x) {
        fill(x);
        under.clear();

        std::vector<std::size_t> col;
        std::vector<double> x_c;

        for (auto &&atom : *this) {
            std::size_t order = 0;
            if (atom.kind() == 0) {
                forEachNeigh(atom, [&](auto const &neigh, double, double,
                                       double, double) {
                    if (neigh.kind() == 0 && bonded(atom, neigh)) {
                        ++order;
                    }
                });
            }
            if (order < 8) {
                under.push_back(atom.pos());
            }
            x_c.push_back(atom.pos()[0]);
            x_c.push_back(atom.pos()[1]);
            x_c.push_back(atom.pos()[2]);
            col.push_back(order);
        }

        for (std::size_t i = 0; i < N; ++i) {
            refineKmeans();
        }

        for (Kmean const &mean : means) {
            x_c.push_back(mean.pos[0]);
            x_c.push_back(mean.pos[1]);
            x_c.push_back(mean.pos[2]);
            col.push_back(1);
            std::cout << mean.pos.transpose() << " : " << mean.size << '\n';
        }

        output(x_c, col);

        return col;
    }
};
