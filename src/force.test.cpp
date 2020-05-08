#include <iostream>

#include "EAM.hpp"
#include "Force.hpp"

int main() {

    // auto eamData = parseTabEAM("/home/cdt1902/dis/CATkS/data/PotentialA.fs");
    //
    // processTab(eamData); // remove factor of r from some potentials
    //
    // numericalDiff(eamData);
    //
    Vector x = {{1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2}};
    //
    std::vector<int> kinds(x.size() / 3, 0);
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
              {1, 0, 3, 0, 3, 0, 3}};

    f(x);

    // std::cout << box.numCells() << std::endl;

    return 0;
}
