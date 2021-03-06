#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "utils.hpp"

template <typename T, typename K>
void dumpXYX(std::string const &file, T const &coords, K const &kinds) {
    CHECK((std::size_t)kinds.size() * 3 == (std::size_t)coords.size(),
          " wrong number of atoms/kinds");

    std::ofstream outfile{file};

    outfile << kinds.size() << std::endl;
    outfile << "This is C.J.Williams' dumpfile";

    outfile << std::setprecision(15);

    for (std::size_t i = 0; i < kinds.size(); ++i) {
        outfile << '\n'
                << kinds[i] << ' ' << coords[3 * i + 0] << ' '
                << coords[3 * i + 1] << ' ' << coords[3 * i + 2];
    }
}

template <typename T, typename K>
void dumpH(std::string const &file, double time, T const &coords,
           K const &kinds) {
    std::ofstream outfile{file, std::ios::app};

    outfile << time;

    for (std::size_t i = 0; i < kinds.size(); ++i) {
        if (kinds[i] == 1) {
            outfile << ' ' << coords[3 * i + 0] << ' ' << coords[3 * i + 1]
                    << ' ' << coords[3 * i + 2];
        }
    }

    outfile << "\n";
}

int FRAME = 0;

static const std::string head{"dump/frame_"};
static const std::string tail{".xyz"};

template <typename T, typename K> void output(T const &x, K const &kinds) {
    dumpXYX(head + std::to_string(FRAME++) + tail, x, kinds);
}

void output(Vector const &x) { output(x, std::vector<int>(x.size() / 3, 0)); }
