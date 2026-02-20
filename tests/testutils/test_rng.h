/**
 * @file test_rng.h
 * @brief Stateless random number generation for LAPACK test infrastructure.
 *
 * Uses xoshiro256+ (Blackman & Vigna, 2018), a fast high-quality PRNG.
 * All functions take an explicit uint64_t state[static 4] parameter —
 * no global state. The caller owns the state (uint64_t[4]), initializes
 * it via rng_seed(), and passes it to every function that needs randomness.
 *
 * Reference: https://prng.di.unimi.it/xoshiro256plus.c
 */

#ifndef TEST_RNG_H
#define TEST_RNG_H

#include <stdint.h>
#include <math.h>
#include "semicolon_lapack/types.h"

/**
 * Rotate left helper.
 */
static inline uint64_t rng_rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/**
 * SplitMix64 - used to initialize xoshiro256+ state from a single seed.
 * Reference: https://prng.di.unimi.it/splitmix64.c
 */
static inline uint64_t rng_splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * Initialize RNG state from a single 64-bit seed.
 *
 * @param[out] state  The 4-element state array to initialize.
 * @param[in]  seed   Any 64-bit value (0 is fine, will be expanded).
 */
static inline void rng_seed(uint64_t state[static 4], uint64_t seed) {
    state[0] = rng_splitmix64(&seed);
    state[1] = rng_splitmix64(&seed);
    state[2] = rng_splitmix64(&seed);
    state[3] = rng_splitmix64(&seed);
}

/**
 * Generate next 64-bit random value.
 * This is the core xoshiro256+ algorithm.
 *
 * @param[in,out] state  The 4-element state array.
 */
static inline uint64_t rng_next(uint64_t state[static 4]) {
    const uint64_t result = state[0] + state[3];
    const uint64_t t = state[1] << 17;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];

    state[2] ^= t;
    state[3] = rng_rotl(state[3], 45);

    return result;
}

/**
 * Generate uniform random f64 in (0, 1).
 * Uses upper 53 bits for full f64 precision mantissa.
 * Excludes exactly 0.0 and 1.0 (matches LAPACK dlaran behavior).
 *
 * @param[in,out] state  The 4-element state array.
 */
static f64 rng_uniform(uint64_t state[static 4]) {
    /* Convert to [0, 1) then shift to (0, 1) */
    f64 u = (rng_next(state) >> 11) * 0x1.0p-53;
    /* Ensure we never return exactly 0 or 1 */
    if (u == 0.0) u = 0x1.0p-53;
    return u;
}

/**
 * Generate uniform random f64 in (-1, 1).
 *
 * @param[in,out] state  The 4-element state array.
 */
static inline f64 rng_uniform_symmetric(uint64_t state[static 4]) {
    return 2.0 * rng_uniform(state) - 1.0;
}

/**
 * Generate standard normal random f64 N(0,1).
 * Uses Box-Muller transform (same as LAPACK dlarnd).
 *
 * @param[in,out] state  The 4-element state array.
 */
static inline f64 rng_normal(uint64_t state[static 4]) {
    static const f64 TWOPI = 6.28318530717958647692528676655900576839;
    f64 u1 = rng_uniform(state);
    f64 u2 = rng_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(TWOPI * u2);
}

/**
 * Generate random value according to distribution type.
 * Compatible with LAPACK IDIST parameter.
 *
 * @param[in,out] state  The 4-element state array.
 * @param[in]     idist  1 = uniform(0,1), 2 = uniform(-1,1), 3 = normal(0,1)
 */
static f64 rng_dist(uint64_t state[static 4], int idist) {
    switch (idist) {
        case 1: return rng_uniform(state);
        case 2: return rng_uniform_symmetric(state);
        case 3: return rng_normal(state);
        default: return rng_uniform(state);
    }
}

/**
 * Fill array with random values from specified distribution.
 *
 * @param[in,out] state  The 4-element state array.
 * @param[in]     idist  Distribution type (1, 2, or 3)
 * @param[in]     n      Number of values to generate
 * @param[out]    x      Output array of dimension n
 */
static inline void rng_fill(uint64_t state[static 4], int idist, int n, f64* x) {
    for (int i = 0; i < n; i++) {
        x[i] = rng_dist(state, idist);
    }
}

#endif /* TEST_RNG_H */
