#pragma once

#include <bitset>
#include <iostream>
#include <memory>

#include "MurmurHash3.h"
#include "nauty.h"
#include "utils.hpp"

class NautyGraph {
  private:
    int n;
    int m;

    std::unique_ptr<graph[]> g;
    std::unique_ptr<graph[]> cg;

    std::unique_ptr<int[]> lab;
    std::unique_ptr<int[]> ptn;
    std::unique_ptr<int[]> orbits;

    DEFAULTOPTIONS_GRAPH(options);

    bool ready = false;

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

    void clear() {
        EMPTYGRAPH(g.get(), m, n);

        ready = false;
    }

    void addEdge(int v, int w) {
        check(v < n && w < n, "out of bounds " << v << ' ' << w);
        check(v >= 0 && w >= 0, "out of bounds " << v << ' ' << w);

        ADDONEEDGE(g.get(), v, w, m);

        ready = false;
    }

    int const *getCanonical() {
        densenauty(g.get(), lab.get(), ptn.get(), orbits.get(), &options,
                   &stats, m, n, cg.get());

        ready = true;

        return lab.get();
    }

    void print() const {
        for (int i = 0; i < n; ++i) {
            std::string str;
            for (int j = 0; j < m; ++j) {
                str += std::bitset<8 * sizeof(graph)>(g[j + m * i]).to_string();
            }
            std::cout << str.substr(0, n) << std::endl;
        }
    }

    void printCanon() const {
        for (int i = 0; i < n; ++i) {
            std::string str;
            for (int j = 0; j < m; ++j) {
                str +=
                    std::bitset<8 * sizeof(graph)>(cg[j + m * i]).to_string();
            }
            std::cout << str.substr(0, n) << std::endl;
        }
    }

    std::array<std::size_t, 2> hash() const {
        static_assert(sizeof(std::size_t) * 8 == 64, "need 64bit std::size_t");

        check(ready, "gotta call nauty first");

        std::array<std::size_t, 2> hash_out;

        MurmurHash3_x86_128(cg.get(), sizeof(graph) * n * m, 0,
                            hash_out.data());

        return hash_out;
    }
};
