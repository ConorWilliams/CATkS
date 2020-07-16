#pragma once

// utilities to pass, prune and differenciate EAM force files

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "Spline.hpp"
#include "utils.hpp"

// struct holds setfl formatted file data;
class TabEAM {
  private:
    std::size_t numS;

    double rCut;

    std::vector<std::size_t> m_atomic;
    std::vector<double> m_mass;

    // EAM potentials

    std::vector<NaturalSpline> m_frho;
    std::vector<NaturalSpline> m_phir;
    std::vector<NaturalSpline> m_vr;

    std::vector<std::size_t> symIdx;

    inline std::size_t toIdx(std::size_t i, std::size_t j) const {
        return i + numS * j;
    }

  public:
    TabEAM(std::size_t numS, double cut)
        : numS(numS), rCut(cut), m_atomic(numS), m_mass(numS), m_frho(numS),
          m_phir(numS * numS), m_vr(numS * (numS + 1) / 2),
          symIdx(numS * numS) {

        for (std::size_t i = 0; i < numS; ++i) {
            for (std::size_t j = 0; j < numS; ++j) {
                symIdx[toIdx(i, j)] = i + (2 * numS - 1 - j) * j / 2;
                std::cout << i << ',' << j << "->" << symIdx[toIdx(i, j)]
                          << '\n';
            }
        }
    }

    inline double rcut() const { return rCut; }

    inline double &getMass(std::size_t i) { return m_mass[i]; }
    inline size_t &getAtomicNum(std::size_t i) { return m_atomic[i]; }

    inline double const &getMass(std::size_t i) const { return m_mass[i]; }
    inline size_t const &getAtomicNum(std::size_t i) const {
        return m_atomic[i];
    }

    inline std::size_t numSpecies() const { return numS; }

    inline NaturalSpline &getF(std::size_t i) { return m_frho[i]; }

    inline NaturalSpline &getP(std::size_t i, std::size_t j) {
        return m_phir[toIdx(i, j)];
    }
    inline NaturalSpline &getV(std::size_t i, std::size_t j) {
        return m_vr[symIdx[toIdx(i, j)]];
    }

    inline NaturalSpline const &getF(std::size_t i) const {
        // std::cout << "getF " << i << '\n';
        return m_frho[i];
    }

    inline NaturalSpline const &getP(std::size_t i, std::size_t j) const {
        // std::cout << "getP " << i << ' ' << j << ' ' << toIdx(i, j) << '\n';
        return m_phir[toIdx(i, j)];
    }
    inline NaturalSpline const &getV(std::size_t i, std::size_t j) const {
        // std::cout << "getV " << i << ' ' << j << '\n';
        return m_vr[symIdx[toIdx(i, j)]];
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

    TabEAM data(numS, cut);

    Vector ftemp{numP};
    Vector ptemp{numR};
    Vector vtemp{numR};

    for (std::size_t i = 0; i < numS; ++i) {
        // read species info
        ensureGetline(file, line);
        // std::cout << line << std::endl;
        // toStream(line) >> data.number[i] >> data.mass[i];

        // read F
        for (std::size_t j = 0; j < numP / 5; ++j) {
            ensureGetline(file, line);
            toStream(line) >> ftemp[j * 5 + 0] >> ftemp[j * 5 + 1] >>
                ftemp[j * 5 + 2] >> ftemp[j * 5 + 3] >> ftemp[j * 5 + 4];
        }

        data.getF(i).computeCoeff(ftemp, delP);

        // read phi

        for (std::size_t j = 0; j < numS; ++j) {
            for (std::size_t k = 0; k < numR / 5; ++k) {
                ensureGetline(file, line);
                toStream(line) >> ptemp[k * 5 + 0] >> ptemp[k * 5 + 1] >>
                    ptemp[k * 5 + 2] >> ptemp[k * 5 + 3] >> ptemp[k * 5 + 4];
            }

            data.getP(i, j).computeCoeff(ptemp, delR);

            // for (std::size_t k = 0; k < numR; ++k) {
            //     std::cout << ptemp[k] << ':' << data.getP(0, 0)(k * delR)
            //               << ' ';
            // }
        }
    }

    // read v ***IMPORTANT: tabulated as r*v ****
    for (std::size_t i = 0; i < numS; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            for (std::size_t k = 0; k < numR / 5; ++k) {
                ensureGetline(file, line);
                toStream(line) >> vtemp[k * 5 + 0] >> vtemp[k * 5 + 1] >>
                    vtemp[k * 5 + 2] >> vtemp[k * 5 + 3] >> vtemp[k * 5 + 4];
            }

            for (std::size_t k = 0; k < numR; ++k) {
                vtemp[k] /= k * delR;
            }

            vtemp[0] = vtemp[1]; // fixup divide by zero

            data.getV(i, j).computeCoeff(vtemp, delR);
        }
    }

    return data;
}
