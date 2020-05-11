#pragma once

#include <fstream>
#include <string>

#include <utils.hpp>

static std::string symbol[2] = {"Fe", "H"};

template <typename K>
void dumpXYX(std::string const &file, Vector const &coords, K kinds) {
    check(kinds.size() * 3 == (std::size_t)coords.size(),
          " wrong number of atoms/kinds");

    std::ofstream outfile{file};

    outfile << kinds.size() << std::endl;
    outfile << "This is C. J. Williams dumpfile";

    for (std::size_t i = 0; i < kinds.size(); ++i) {
        outfile << '\n'
                << symbol[kinds[i]] << ' ' << coords[3 * i + 0] << ' '
                << coords[3 * i + 1] << ' ' << coords[3 * i + 2];
    }
}
