#pragma once

#include <algorithm>
#include <vector>

#include "Cell.hpp"

void cellSort(Vector &x, std::vector<int> &kinds, Box const &box) {
    CHECK((size_t)x.size() / 3 == kinds.size(), "atom mismatch");

    CellList<AtomBase> cellList{box, kinds};

    cellList.fillList(x);

    std::sort(cellList.begin(), cellList.end(),
              [&](AtomBase const &a, AtomBase const &b) -> bool {
                  return box.lambda(a) < box.lambda(b);
              });

    for (std::size_t i = 0; i < cellList.size(); ++i) {
        x[3 * i + 0] = cellList[i][0];
        x[3 * i + 1] = cellList[i][1];
        x[3 * i + 2] = cellList[i][2];

        kinds[i] = cellList[i].kind();
    }
}
