#include <iostream>

#include "Array.hpp"

#include "EAM.hpp"

struct A {};
struct B : public A {};

template <typename T, size_t N> int foo(Array_d<T, N>) { return 3; }
int foo(A) { return 1; }

template <typename T, size_t N> __global__ void MyKernel(Array_d<T, N> arr) {
    arr(threadIdx.x, threadIdx.y) *= max(2, (int)arr(threadIdx.x, threadIdx.y));
}

int main() {

    // ArrayGPU<float, 2> arrG;

    constexpr std::size_t n = 46;

    Array_h<float, 2> arr(n, n / 2);

    for (std::size_t j = 0; j < n / 2; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            arr(i, j) = i + n * j;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n / 2; ++j) {
            std::cout << arr(i, j) << ' ';
        }
        std::cout << '\n';
    }

    arr.hostToDevice();

    MyKernel<<<1, {n, n / 2}>>>(arr);

    arr.deviceToHost();

    cudaDeviceSynchronize();

    // arrG(1, 1) = arrC(2, 2);

    std::cout << "passing " << foo(B{}) << '\n';
    std::cout << "passing " << foo(arr) << '\n';

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n / 2; ++j) {
            std::cout << arr(i, j) << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "ALL GOOD\n";

    return 0;
}
