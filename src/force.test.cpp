#include <iostream>

#include "EAM.hpp"
#include "Forces.hpp"

int main() {
    enum : uint8_t { Fe = 0, H = 1 };

    // auto eamData = parseTabEAM("/home/cdt1902/dis/CATkS/data/PotentialA.fs");
    //
    // processTab(eamData); // remove factor of r from some potentials
    //
    // numericalDiff(eamData);
    //

    constexpr double LAT = 2.855700;

    Vector x(7 * 7 * 7 * 3 * 2);

    Vector grad(x.size());

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

    std::vector<int> kinds(x.size() / 3, Fe);
    //
    // // std::cout << lam.numCells() << " " << lam(-1, -1, -1) << std::endl;
    //
    // std::cout << std::fmod(12.01, 3) << std::endl;
    //
    // Box box{1, -5, 5, -5, 5, -5, 5};
    //
    // LinkedCellList lcl{kinds, box};
    //
    // lcl.makeCellList(x);

    FuncEAM f{"/home/cdt1902/dis/CATkS/data/PotentialA.fs",
              kinds,
              0,
              7 * LAT,
              0,
              7 * LAT,
              0,
              7 * LAT};

    x[0] += 0.1;

    f(x, grad);

    std::cout << x.size() << ' ' << x[1] << ' ' << x[2] << std::endl;

    std::cout << grad[0] << ' ' << grad[1] << ' ' << grad[2] << std::endl;

    // f(x, grad);

    // std::cout << box.numCells() << std::endl;

    return 0;
}
