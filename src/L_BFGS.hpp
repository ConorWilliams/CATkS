#pragma once

#include <cstddef> // std::size_t
#include <iostream>

#include "Eigen/Core"

using Vector = Eigen::Array<double, Eigen::Dynamic, 1>;

template <long M> class CoreLBFGS {
  private:
    static_assert(M > 0, "Invalid amount of history");

    Eigen::Array<double, Eigen::Dynamic, M> m_s;
    Eigen::Array<double, Eigen::Dynamic, M> m_y;
    Eigen::Array<double, Eigen::Dynamic, M> m_rho;

    Vector m_prev_pos;
    Vector m_prev_grad;

    Vector m_a;

    std::size_t m_dims;
    long m_k;

  public:
    CoreLBFGS(std::size_t dims)
        : m_s{dims, M}, m_y{dims, M}, m_rho{dims, M}, m_prev_pos{dims},
          m_prev_grad{dims}, m_a{M}, m_dims{dims}, m_k{0} {}

    inline void clear() { m_k = 0; }

    /**
     * Computes the H_k g_k product using the L-BFGS two-loop recursion
     *
     * param q      p_k vector to write output to
     * param pos    x_k current coordinate
     * param grad   g-k current gradient
     */
    inline void operator()(Vector &q, Vector const &pos, Vector const &grad) {
        q = grad;

        long idx = (m_k - 1) % M;

        // compute k-1 th y,s,rho
        if (m_k > 0) {
            m_s.col(idx) = pos - m_prev_pos;
            m_y.col(idx) = grad - m_prev_grad;
            m_rho(idx) = 1 / (m_s.col(idx) * m_y.col(idx)).sum();
        }

        long incr = m_k - M;
        long bound = M;

        if (m_k <= M) {
            incr = 0;
            bound = m_k;
        }

        // loop 1
        for (long i = bound - 1; i >= 0; --i) {
            long j = (i + incr) % M;
            // std::cout << "one: " << j << std::endl;

            m_a(j) = m_rho(j) * (m_s.col(j) * q).sum();
            q += m_a(j) * m_y.col(j);
        }

        // scaling
        if (m_k > 0) {
            q *= 1 / (m_rho(idx) * (m_y.col(idx) * m_y.col(idx)).sum());
        }

        // loop 2
        for (long i = 0; i <= bound - 1; ++i) {
            long j = (i + incr) % M;
            // std::cout << "two: " << j << std::endl;

            double b = m_rho(j) * (m_y.col(j) * q).sum();
            q += (m_a(j) - b) * m_s.col(j);
        }

        m_prev_pos = pos;
        m_prev_grad = grad;

        ++m_k;

        return;
    }

    void dump() { std::cout << m_s << std::endl; }

    CoreLBFGS(const CoreLBFGS &other) = default;
    CoreLBFGS(CoreLBFGS &&other) = default;
    CoreLBFGS &operator=(const CoreLBFGS &other) = default;
    CoreLBFGS &operator=(CoreLBFGS &&other) = default;
};
