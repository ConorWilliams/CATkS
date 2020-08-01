#pragma once

#include <algorithm>
#include <vector>

#include "Cell.hpp"

void cellSort(Vector &x, std::vector<int> &kinds, Box const &box) {
    CHECK((size_t)x.size() / 3 == kinds.size(), "atom mismatch");

    CellListSorted<AtomSortBase> cellList{box, kinds};

    cellList.fill(x);

    for (std::size_t i = 0; i < cellList.size(); ++i) {
        auto j = cellList[i].index();

        x[3 * j + 0] = cellList[i][0];
        x[3 * j + 1] = cellList[i][1];
        x[3 * j + 2] = cellList[i][2];

        kinds[j] = cellList[i].kind();
    }
}
