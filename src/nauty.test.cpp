#include "Nauty.hpp"

/* MAXN=0 is defined by nauty.h, which implies dynamic allocation */
int main(int argc, char *argv[]) {

    int nn = 4;

    NautyGraph tmp{nn};

    tmp.addEdge(0, 1);
    tmp.addEdge(1, 2);
    tmp.addEdge(2, 3);
    tmp.addEdge(3, 0);

    tmp.print();

    int const *ptr = tmp.getCanonical();

    tmp.printCanon();

    for (int i = 0; i < nn; ++i) {
        std::cout << ptr[i] << std::endl;
    }

    return 0;
}
