#pragma once

#include <array>
#include <functional>

#include "MurmurHash3.h"
#include "utils.hpp"

using sint_t = uint8_t;
inline constexpr sint_t RADIAL_BINS = 32; // power of 2 for fast division
inline constexpr double BIN_WIDTH = 1. / RADIAL_BINS;

class Rdf {
  private:
    std::array<sint_t, RADIAL_BINS> rdf{};

  public:
    Rdf() = default;

    inline void add(double r) {
        check(r >= 0 && r < 1, "not on unit interval");

        std::size_t bin = r * RADIAL_BINS;
        ++rdf[bin];

        // bin fuzzing
        if ((r + BIN_WIDTH / 4) * RADIAL_BINS != bin) {
            ++rdf[bin + 1];
        } else if ((r - BIN_WIDTH / 4) * RADIAL_BINS != bin) {
            ++rdf[bin - 1];
        }
    }

    inline sint_t const *data() const { return rdf.data(); }

    friend inline bool operator==(Rdf a, Rdf b) { return a.rdf == b.rdf; }
};

// specialising for std::unordered set
namespace std {
template <> struct hash<Rdf> {
    std::size_t operator()(Rdf const &x) const {
        static_assert(sizeof(std::size_t) * 8 == 64, "need 64bit std::size_t");

        std::size_t hash_out[2];
        MurmurHash3_x86_128(x.data(), sizeof(x), 0, hash_out);
        return hash_out[0] ^ hash_out[1];
    }
};
} // namespace std
