#pragma once

// utilities to pass, prune and differenciate EAM force files

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "utils.hpp"

// struct holds setfl formatted file data;
struct TabEAM {
    std::size_t numSpecies;
    std::size_t numPntsP; // P -> density coord
    std::size_t numPntsR; // R -> distance coord

    double deltaR;
    double deltaP;

    double rCut;

    Eigen::ArrayXi number;
    Eigen::ArrayXd mass;

    // EAM potentials
    Eigen::ArrayXXd tabF; // one column per atom
    Eigen::Array<Eigen::ArrayXd, Eigen::Dynamic, Eigen::Dynamic> tabPhi;
    Eigen::Array<Eigen::ArrayXd, Eigen::Dynamic, Eigen::Dynamic> tabV;

    // derivatives of above
    Eigen::ArrayXXd difF; // one column per atom
    Eigen::Array<Eigen::ArrayXd, Eigen::Dynamic, Eigen::Dynamic> difPhi;
    Eigen::Array<Eigen::ArrayXd, Eigen::Dynamic, Eigen::Dynamic> difV;

    std::size_t rToIndex(double r) {
        check(r > 0 && r <= numPntsP * deltaR, "r outside boundary");
        return r / deltaR;
    }

    std::size_t pToIndex(double p) {
        check(p > 0 && p <= numPntsP * deltaP, "p outside boundary");
        return p / deltaP;
    }

    TabEAM(std::size_t numS, std::size_t numP, std::size_t numR, double delP,
           double delR, double cut)
        : numSpecies(numS), numPntsP(numP), numPntsR(numR), deltaR(delR),
          deltaP(delP), rCut(cut) {

        number.resize(numSpecies);
        mass.resize(numSpecies);

        tabF.resize(numPntsP, numSpecies);
        tabPhi.resize(numSpecies, numSpecies);
        tabV.resize(numSpecies, numSpecies);

        difF.resize(numPntsP, numSpecies);
        difPhi.resize(numSpecies, numSpecies);
        difV.resize(numSpecies, numSpecies);

        for (std::size_t i = 0; i < numSpecies; ++i) {
            for (std::size_t j = 0; j < numSpecies; ++j) {
                tabPhi(i, j).resize(numPntsR);
                tabV(i, j).resize(numPntsR);

                difPhi(i, j).resize(numPntsR);
                difV(i, j).resize(numPntsR);
            }
        }
    }
};

std::istringstream toStream(std::string const &line) {
    return std::istringstream{line};
}

void ensureGetline(std::ifstream &file, std::string &line) {
    if (!std::getline(file, line)) {
        throw std::runtime_error("file terminated too soon");
    }
}

// read raw tabulated data in SETFL file fmt to stuct^
TabEAM parseTabEAM(std::string const &fileName) {
    std::ifstream file(fileName);
    std::string line;

    std::size_t numS, numP, numR;
    double delP, delR, cut;

    // skip to 4th line
    for (int i = 0; i < 4; ++i) {
        ensureGetline(file, line);
    }
    toStream(line) >> numS;

    ensureGetline(file, line);
    toStream(line) >> numP >> delP >> numR >> delR >> cut;

    TabEAM data{numS, numP, numR, delP, delR, cut};

    for (std::size_t i = 0; i < data.numSpecies; ++i) {
        // read species info
        ensureGetline(file, line);
        // std::cout << line << std::endl;
        toStream(line) >> data.number[i] >> data.mass[i];

        // read F
        for (std::size_t j = 0; j < data.numPntsP / 5; ++j) {
            ensureGetline(file, line);
            toStream(line) >> data.tabF.col(i)[j * 5 + 0] >>
                data.tabF.col(i)[j * 5 + 1] >> data.tabF.col(i)[j * 5 + 2] >>
                data.tabF.col(i)[j * 5 + 3] >> data.tabF.col(i)[j * 5 + 4];
        }

        // read phi
        for (std::size_t j = 0; j < data.numSpecies; ++j) {
            for (std::size_t k = 0; k < data.numPntsR / 5; ++k) {
                ensureGetline(file, line);
                toStream(line) >> data.tabPhi(i, j)[k * 5 + 0] >>
                    data.tabPhi(i, j)[k * 5 + 1] >>
                    data.tabPhi(i, j)[k * 5 + 2] >>
                    data.tabPhi(i, j)[k * 5 + 3] >>
                    data.tabPhi(i, j)[k * 5 + 4];
            }
        }
    }

    // read v ***IMPORTANT: tabulated as r*v ****
    for (std::size_t i = 0; i < data.numSpecies; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            for (std::size_t k = 0; k < data.numPntsR / 5; ++k) {
                ensureGetline(file, line);
                toStream(line) >> data.tabV(i, j)[k * 5 + 0] >>
                    data.tabV(i, j)[k * 5 + 1] >> data.tabV(i, j)[k * 5 + 2] >>
                    data.tabV(i, j)[k * 5 + 3] >> data.tabV(i, j)[k * 5 + 4];
            }
        }
    }

    // make V symetric
    for (std::size_t i = 0; i < data.numSpecies; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            data.tabV(j, i) = data.tabV(i, j);
        }
    }

    return data;
}

// removes factor of r from tabulated potentials
void processTab(TabEAM &data) {
    for (std::size_t i = 0; i < data.numSpecies; ++i) {
        for (std::size_t j = 0; j < data.numSpecies; ++j) {
            for (std::size_t k = 0; k < data.numPntsR; ++k) {
                data.tabV(i, j)[k] /= k * data.deltaR;
            }
        }
    }
}

template <typename T, typename U>
void differenciate(T const &in, U &&out, double del) {
    std::size_t len = in.size();

    out[0] = (in[1] - in[0]) / del;
    out[len - 1] = (in[len - 1] - in[len - 2]) / del;

    for (std::size_t i = 1; i < len - 1; ++i) {
        out[i] = (in[i + 1] - in[i - 1]) / (2 * del);
    }
}

void numericalDiff(TabEAM &data) {
    for (std::size_t i = 0; i < data.numSpecies; ++i) {
        differenciate(data.tabF.col(i), data.difF.col(i), data.deltaP);
        for (std::size_t j = 0; j < data.numSpecies; ++j) {
            differenciate(data.tabV(i, j), data.difV(i, j), data.deltaR);
            differenciate(data.tabPhi(i, j), data.difPhi(i, j), data.deltaR);
        }
    }
}
