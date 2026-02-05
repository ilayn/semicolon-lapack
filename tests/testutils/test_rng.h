/**
 * @file test_rng.h
 * @brief Random number generator for LAPACK test infrastructure.
 *
 * Uses xoshiro256+ (Blackman & Vigna, 2018), a fast high-quality PRNG.
 * This replaces LAPACK's archaic 48-bit LCG with 4-element seed arrays.
 *
 * Reference: https://prng.di.unimi.it/xoshiro256plus.c
 */

#ifndef TEST_RNG_H
#define TEST_RNG_H

#include <stdint.h>
#include <math.h>

/* xoshiro256+ state - 256 bits */
static uint64_t rng_state[4];

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
static inline uint64_t rng_splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * Initialize RNG with a single 64-bit seed.
 *
 * @param seed  Any 64-bit value (0 is fine, will be expanded)
 */
static inline void rng_seed(uint64_t seed) {
    rng_state[0] = rng_splitmix64(&seed);
    rng_state[1] = rng_splitmix64(&seed);
    rng_state[2] = rng_splitmix64(&seed);
    rng_state[3] = rng_splitmix64(&seed);
}

/**
 * Generate next 64-bit random value.
 * This is the core xoshiro256+ algorithm.
 */
static inline uint64_t rng_next(void) {
    const uint64_t result = rng_state[0] + rng_state[3];
    const uint64_t t = rng_state[1] << 17;

    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];

    rng_state[2] ^= t;
    rng_state[3] = rng_rotl(rng_state[3], 45);

    return result;
}

/**
 * Generate uniform random double in (0, 1).
 * Uses upper 53 bits for full double precision mantissa.
 * Excludes exactly 0.0 and 1.0 (matches LAPACK dlaran behavior).
 */
static inline double rng_uniform(void) {
    /* Convert to [0, 1) then shift to (0, 1) */
    double u = (rng_next() >> 11) * 0x1.0p-53;
    /* Ensure we never return exactly 0 or 1 */
    if (u == 0.0) u = 0x1.0p-53;
    return u;
}

/**
 * Generate uniform random double in (-1, 1).
 */
static inline double rng_uniform_symmetric(void) {
    return 2.0 * rng_uniform() - 1.0;
}

/**
 * Generate standard normal random double N(0,1).
 * Uses Box-Muller transform (same as LAPACK dlarnd).
 */
static inline double rng_normal(void) {
    static const double TWOPI = 6.28318530717958647692528676655900576839;
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    return sqrt(-2.0 * log(u1)) * cos(TWOPI * u2);
}

/**
 * Generate random value according to distribution type.
 * Compatible with LAPACK IDIST parameter.
 *
 * @param idist  1 = uniform(0,1), 2 = uniform(-1,1), 3 = normal(0,1)
 */
static double rng_dist(int idist) {
    switch (idist) {
        case 1: return rng_uniform();
        case 2: return rng_uniform_symmetric();
        case 3: return rng_normal();
        default: return rng_uniform();
    }
}

/**
 * Fill array with random values from specified distribution.
 *
 * @param idist  Distribution type (1, 2, or 3)
 * @param n      Number of values to generate
 * @param x      Output array of dimension n
 */
static inline void rng_fill(int idist, int n, double *x) {
    for (int i = 0; i < n; i++) {
        x[i] = rng_dist(idist);
    }
}

#endif /* TEST_RNG_H */
