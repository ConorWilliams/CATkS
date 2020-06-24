#pragma once

#include <bitset>
#include <iostream>
#include <memory>

/* This program prints generators for the automorphism group of an
n-vertex polygon, where n is a number supplied by the user.
This version uses dynamic allocation.
*/
#include "nauty.h"

class NautyGraph {
  public:
    int n;
    int m;

    std::unique_ptr<graph[]> g;
    std::unique_ptr<graph[]> cg;

    std::unique_ptr<int[]> lab;
    std::unique_ptr<int[]> ptn;
    std::unique_ptr<int[]> orbits;

    DEFAULTOPTIONS_GRAPH(options);

    statsblk stats;

  public:
    NautyGraph(int n_)
        : n{n_}, m{SETWORDSNEEDED(n)}, g{new graph[n * m]{}},
          cg{new graph[n * m]{}}, lab{new int[n]{}}, ptn{new int[n]{}},
          orbits{new int[n]{}} {

        nauty_check(WORDSIZE, m, n, NAUTYVERSIONID); // optional

        options.getcanon = true;
        options.defaultptn = true; // change for coloured graphs
    }

    void clear() { EMPTYGRAPH(g.get(), m, n); }

    void addEdge(std::size_t v, std::size_t w) { ADDONEEDGE(g.get(), v, w, m) }

    int const *getCanonical() {
        densenauty(g.get(), lab.get(), ptn.get(), orbits.get(), &options,
                   &stats, m, n, cg.get());

        return lab.get();
    }

    void print() {
        for (int i = 0; i < n; ++i) {
            std::string str;
            for (int j = 0; j < m; ++j) {
                str += std::bitset<8 * sizeof(graph)>(g[j + m * i]).to_string();
            }
            std::cout << str.substr(0, n) << std::endl;
        }
    }

    void printCanon() {
        for (int i = 0; i < n; ++i) {
            std::string str;
            for (int j = 0; j < m; ++j) {
                str +=
                    std::bitset<8 * sizeof(graph)>(cg[j + m * i]).to_string();
            }
            std::cout << str.substr(0, n) << std::endl;
        }
    }
};
