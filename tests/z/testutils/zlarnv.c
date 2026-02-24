/**
 * @file zlarnv.c
 * @brief Test utility versions of ZLARND/ZLARNV that use xoshiro256+ RNG.
 *
 * These are test-only versions that use our modern xoshiro256+ RNG
 * with explicit state passing instead of LAPACK's 48-bit LCG with
 * 4-element integer seed array.
 *
 * Named zlarnd_rng/zlarnv_rng to avoid conflict with the library's
 * zlarnv which uses the LAPACK seed convention.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlarnd.f and SRC/zlarnv.f
 */

#include <math.h>
#include "verify.h"
#include "test_rng.h"

static const f64 TWOPI = 6.28318530717958647692528676655900576839;

/**
 * zlarnd_rng returns a random complex number from a uniform or normal
 * distribution using xoshiro256+ RNG.
 *
 * @param[in] idist
 *     Specifies the distribution of the random numbers:
 *     = 1:  real and imaginary parts each uniform (0,1)
 *     = 2:  real and imaginary parts each uniform (-1,1)
 *     = 3:  real and imaginary parts each normal (0,1)
 *     = 4:  uniformly distributed on the disc abs(z) <= 1
 *     = 5:  uniformly distributed on the circle abs(z) = 1
 *
 * @param[in,out] state
 *     The 4-element RNG state array, advanced on exit.
 *
 * @return A random complex number from the specified distribution.
 */
c128 zlarnd_rng(const INT idist, uint64_t state[static 4])
{
    f64 t1 = rng_uniform(state);
    f64 t2 = rng_uniform(state);

    if (idist == 1) {
        /* real and imaginary parts each uniform (0,1) */
        return CMPLX(t1, t2);
    } else if (idist == 2) {
        /* real and imaginary parts each uniform (-1,1) */
        return CMPLX(2.0 * t1 - 1.0, 2.0 * t2 - 1.0);
    } else if (idist == 3) {
        /* real and imaginary parts each normal (0,1) */
        f64 r = sqrt(-2.0 * log(t1));
        f64 theta = TWOPI * t2;
        return CMPLX(r * cos(theta), r * sin(theta));
    } else if (idist == 4) {
        /* uniform distribution on the unit disc abs(z) <= 1 */
        f64 r = sqrt(t1);
        f64 theta = TWOPI * t2;
        return CMPLX(r * cos(theta), r * sin(theta));
    } else if (idist == 5) {
        /* uniform distribution on the unit circle abs(z) = 1 */
        f64 theta = TWOPI * t2;
        return CMPLX(cos(theta), sin(theta));
    }
    return CMPLX(0.0, 0.0);
}

/**
 * zlarnv_rng returns a vector of n random complex numbers from a uniform
 * or normal distribution using xoshiro256+ RNG.
 *
 * @param[in] idist
 *     Specifies the distribution of the random numbers:
 *     = 1:  real and imaginary parts each uniform (0,1)
 *     = 2:  real and imaginary parts each uniform (-1,1)
 *     = 3:  real and imaginary parts each normal (0,1)
 *     = 4:  uniformly distributed on the disc abs(z) <= 1
 *     = 5:  uniformly distributed on the circle abs(z) = 1
 *
 * @param[in] n
 *     The number of random numbers to be generated.
 *
 * @param[out] x
 *     The generated random complex numbers.
 *
 * @param[in,out] state
 *     The 4-element RNG state array, advanced on exit.
 */
void zlarnv_rng(const INT idist, const INT n, c128* x,
                uint64_t state[static 4])
{
    if (n <= 0) {
        return;
    }

    for (INT i = 0; i < n; i++) {
        x[i] = zlarnd_rng(idist, state);
    }
}
