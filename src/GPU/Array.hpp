#pragma once

#include <bits/c++config.h>
#include <cuda.h>

// Provides indexed array access to davice/host memory, does not manage memory,
// should only be copy constucted from a derived class that manages memory.
template <typename T, std::size_t N> class Array_d {
  private:
    std::size_t m_stride[N];

    template <typename... Args>
    __host__ __device__ inline std::size_t index(Args... args) const {
        static_assert(sizeof...(Args) == N, "Wrong number of arguments to ()");
        std::size_t const tmp[N] = {args...};
        std::size_t sum = tmp[0];
        for (std::size_t i = 1; i < N; ++i) {
            sum += m_stride[i - 1] * tmp[i];
        }
        return sum;
    }

  protected:
    T *m_data_h;
    T *m_data_d;

    template <typename... Args>
    explicit Array_d(Args... args) : m_stride{args...} {
        static_assert(sizeof...(Args) == N, "Wrong number of arguments @ ctor");
        for (std::size_t i = 1; i < N; ++i) {
            m_stride[i] *= m_stride[i - 1];
        }
    };

  public:
    Array_d() = delete;

    inline std::size_t size() const { return m_stride[N - 1]; }

    template <typename... Args>
    __host__ __device__ inline T &operator()(Args... args) {
#ifdef __CUDA_ARCH__
        return m_data_d[index(args...)];
#else
        return m_data_h[index(args...)];
#endif
    }

    template <typename... Args>
    __host__ __device__ inline T const &operator()(Args... args) const {
#ifdef __CUDA_ARCH__
        return m_data_d[index(args...)];
#else
        return m_data_h[index(args...)];
#endif
    }
};

// CPU wrapper over mulidimenional array, manages memory
template <typename T, std::size_t N> class Array_h : public Array_d<T, N> {
  private:
    using Array_d<T, N>::m_data_d;
    using Array_d<T, N>::m_data_h;

  public:
    using Array_d<T, N>::size;

    template <typename... Args>
    Array_h(Args... args) : Array_d<T, N>::Array_d(args...) {
        //
        m_data_h = static_cast<T *>(malloc(this->size() * sizeof(T)));
        cudaMalloc(&m_data_d, size() * sizeof(T));
    }

    ~Array_h() {
        free(m_data_h);
        cudaFree(m_data_d);
    }

    void hostToDevice() {
        cudaMemcpy(m_data_d, m_data_h, size() * sizeof(T),
                   cudaMemcpyHostToDevice);
    }

    void deviceToHost() {
        cudaMemcpy(m_data_h, m_data_d, size() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }
};

//
// template <typename T> class Array_d;
// template <typename T> class Array_h;
//
// template <typename T> class Array_h {
//   protected:
//     T *m_data;
//     std::size_t m_size;
//
//   public:
//     Array_h(std::size_t size)
//         : m_data{static_cast<T *>(malloc(sizeof(T) * size))},
//         m_size{size} {}
//
//     Array_h(Array_h const &) = delete;
//     Array_h &operator=(Array_h const &) = delete;
//
//     ~Array_h() { free(m_data); }
//
//     Array_h &operator=(Array_d<T> const &other) {
//         cudaMemcpy(data(), other.data(), sizeof(T) * m_size,
//                    cudaMemcpyDeviceToHost);
//     }
//
//     T *data() { return m_data; }
//     T const *data() const { return m_data; }
//     std::size_t size() const { return m_size; }
// };
//
// template <typename T> class Array_d {
//   private:
//     T *m_data;
//     std::size_t m_size;
//
//   public:
//     Array_d(std::size_t size) : m_size{size} {
//         cudaMalloc(&m_data, sizeof(T) * size);
//     }
//
//     Array_d(Array_d const &) = delete;
//     Array_d &operator=(Array_d const &) = delete;
//
//     ~Array_d() { cudaFree(m_data); }
//
//     Array_d &operator=(Array_h<T> const &other) {
//         cudaMemcpy(data(), other.data(), sizeof(T) * m_size,
//                    cudaMemcpyHostToDevice);
//
//         return *this;
//     }
//
//     std::size_t size() const { return m_size; }
//
//     T *data() { return m_data; }
//     T const *data() const { return m_data; }
// };
//
// template <typename T> struct ArrayView : public Array_h<T> {
//     ArrayView(Array_h<T> const &) {}
//     ~ArrayView() {}
// };
