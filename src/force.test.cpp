#include <iostream>

#include "EAM.hpp"
#include "Force2.hpp"

int main() {
    enum : uint8_t { Fe = 0, H = 1 };

    // auto eamData = parseTabEAM("/home/cdt1902/dis/CATkS/data/PotentialA.fs");
    //
    // processTab(eamData); // remove factor of r from some potentials
    //
    // numericalDiff(eamData);
    //

    constexpr double LAT = 2.855700;

    Vector x = {{0, 0, 0, 2.87, 0, 0}};

    Vector grad(x.size());

    cell = 0;

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            for (int k = 0; k < 7; ++k) {
                /* code */
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

    CompEAM f{"/home/cdt1902/dis/CATkS/data/PotentialA.fs",
              kinds,
              0,
              7 * LAT,
              0,
              7 * LAT,
              0,
              7 * LAT};

    f(x, grad);

    std::cout << grad.transpose() << std::endl;

    // f(x, grad);

    // std::cout << box.numCells() << std::endl;

    return 0;
}
