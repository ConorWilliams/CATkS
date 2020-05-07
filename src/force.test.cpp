#include <iostream>

#include "EAM.hpp"
#include "Force.hpp"

int main() {

    auto eamData = parseTabEAM("/home/cdt1902/dis/CATkS/data/PotentialA.fs");

    processTab(eamData); // remove factor of r from some potentials

    numericalDiff(eamData);

    std::vector<int> kinds = {0, 1};

    Vector x = {{1, 1, 1, 1, 1, 0.1}};

    // std::cout << lam.numCells() << " " << lam(-1, -1, -1) << std::endl;

    Box box{1, 0, 3, 0, 3, 0, 3};

    LinkedCellList lcl{kinds, box};

    lcl.makeCellList(x);

    std::cout << box.numCells() << std::endl;

    return 0;
}
