#pragma once

#include <bits/c++config.h>
#include <cuda.h>

template <typename T> class ArrayGPU;
template <typename T> class ArrayCPU;

template <typename T> class ArrayCPU {
  protected:
    T *m_data;
    std::size_t m_size;

  public:
    ArrayCPU(std::size_t size)
        : m_data{static_cast<T *>(malloc(sizeof(T) * size))}, m_size{size} {}

    ArrayCPU(ArrayCPU const &) = delete;
    ArrayCPU &operator=(ArrayCPU const &) = delete;

    ~ArrayCPU() { free(m_data); }

    ArrayCPU &operator=(ArrayGPU<T> const &other) {
        cudaMemcpy(data(), other.data(), sizeof(T) * m_size,
                   cudaMemcpyDeviceToHost);
    }

    T *data() { return m_data; }
    T const *data() const { return m_data; }
    std::size_t size() const { return m_size; }
};

template <typename T> class ArrayGPU {
  private:
    T *m_data;
    std::size_t m_size;

  public:
    ArrayGPU(std::size_t size) : m_size{size} {
        cudaMalloc(&m_data, sizeof(T) * size);
    }

    ArrayGPU(ArrayGPU const &) = delete;
    ArrayGPU &operator=(ArrayGPU const &) = delete;

    ~ArrayGPU() { cudaFree(m_data); }

    ArrayGPU &operator=(ArrayCPU<T> const &other) {
        cudaMemcpy(data(), other.data(), sizeof(T) * m_size,
                   cudaMemcpyHostToDevice);

        return *this;
    }

    std::size_t size() const { return m_size; }

    T *data() { return m_data; }
    T const *data() const { return m_data; }
};

template <typename T> struct ArrayView : public ArrayCPU<T> {
    ArrayView(ArrayCPU<T> const &) {}
    ~ArrayView() {}
};
