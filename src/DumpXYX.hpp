#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "utils.hpp"

template <typename K>
void dumpXYX(std::string const &file, Vector const &coords, K kinds) {
    check(kinds.size() * 3 == (std::size_t)coords.size(),
          " wrong number of atoms/kinds");

    std::ofstream outfile{file};

    outfile << kinds.size() << std::endl;
    outfile << "This is C.J.Williams' dumpfile";

    for (std::size_t i = 0; i < kinds.size(); ++i) {
        outfile << '\n'
                << kinds[i] << ' ' << coords[3 * i + 0] << ' '
                << coords[3 * i + 1] << ' ' << coords[3 * i + 2];
    }
}

int FRAME = 0;
static const std::string head{"/home/cdt1902/dis/CATkS/plt/dump/all_"};
static const std::string tail{".xyz"};

template <typename T> void output(Vector const &x, T const &kinds) {
    dumpXYX(head + std::to_string(FRAME++) + tail, x, kinds);
}

void output(Vector const &x) { output(x, std::vector<int>(x.size() / 3, 0)); }
