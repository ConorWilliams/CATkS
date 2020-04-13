#pragma once

#include <cstddef> // std::size_t
#include <iostream>

#include "Eigen/Core"

using Vector = Eigen::Array<double, Eigen::Dynamic, 1>;

template <std::size_t M> class CoreLBFGS {
  private:
    Eigen::Array<double, Eigen::Dynamic, M> m_s;
    Eigen::Array<double, Eigen::Dynamic, M> m_y;
    Eigen::Array<double, Eigen::Dynamic, M> m_rho;

    Vector m_a;

    std::size_t m_dims;
    std::size_t k;

  public:
    CoreLBFGS(std::size_t dims)
        : m_s{dims, M}, m_y{dims, M}, m_rho{dims, M}, m_a{dims}, m_dims{dims},
          k{0} {}

    Vector step(Vector const &pos, Vector const &grad) {
        Vector q = grad;

        q = pos + grad;

        return q;
    }

    void dump() { std::cout << m_s << std::endl; }

    CoreLBFGS(const CoreLBFGS &other) = default;
    CoreLBFGS(CoreLBFGS &&other) = default;
    CoreLBFGS &operator=(const CoreLBFGS &other) = default;
    CoreLBFGS &operator=(CoreLBFGS &&other) = default;
};
