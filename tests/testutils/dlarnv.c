/**
 * @file dlarnv.c
 * @brief Test utility version of DLARNV that uses xoshiro256+ RNG.
 *
 * These are test-only versions that use our modern xoshiro256+ RNG
 * with explicit state passing instead of LAPACK's 48-bit LCG with
 * 4-element integer seed array.
 *
 * Named dlarnv_rng/dlaran_rng to avoid conflict with the library's
 * dlarnv/dlaran which use the LAPACK seed convention.
 */

#include <math.h>
#include "verify.h"
#include "test_rng.h"

/**
 * dlarnv_rng returns a vector of n random real numbers from a uniform or
 * normal distribution using xoshiro256+ RNG.
 *
 * @param[in] idist
 *     Specifies the distribution of the random numbers:
 *     = 1:  uniform (0,1)
 *     = 2:  uniform (-1,1)
 *     = 3:  normal (0,1)
 *
 * @param[in] n
 *     The number of random numbers to be generated.
 *
 * @param[out] x
 *     The generated random numbers.
 *
 * @param[in,out] state
 *     The 4-element RNG state array, advanced on exit.
 */
void dlarnv_rng(const int idist, const int n, f64* x,
                uint64_t state[static 4])
{
    if (n <= 0) {
        return;
    }

    for (int i = 0; i < n; i++) {
        x[i] = rng_dist(state, idist);
    }
}

/**
 * dlaran_rng returns a random real number from a uniform (0,1) distribution
 * using xoshiro256+ RNG.
 *
 * @param[in,out] state
 *     The 4-element RNG state array, advanced on exit.
 *
 * @return A random real number in (0,1).
 */
f64 dlaran_rng(uint64_t state[static 4])
{
    return rng_uniform(state);
}
