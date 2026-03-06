/**
 * @file clarnv.c
 * @brief Test utility versions of ZLARND/CLARNV that use xoshiro256+ RNG.
 *
 * These are test-only versions that use our modern xoshiro256+ RNG
 * with explicit state passing instead of LAPACK's 48-bit LCG with
 * 4-element integer seed array.
 *
 * Named clarnd_rng/clarnv_rng to avoid conflict with the library's
 * clarnv which uses the LAPACK seed convention.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlarnd.f and SRC/clarnv.f
 */

#include <math.h>
#include "verify.h"
#include "test_rng.h"

static const f32 TWOPI = 6.28318530717958647692528676655900576839f;

/**
 * clarnd_rng returns a random complex number from a uniform or normal
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
c64 clarnd_rng(const INT idist, uint64_t state[static 4])
{
    f32 t1 = rng_uniform_f32(state);
    f32 t2 = rng_uniform_f32(state);

    if (idist == 1) {
        /* real and imaginary parts each uniform (0,1) */
        return CMPLXF(t1, t2);
    } else if (idist == 2) {
        /* real and imaginary parts each uniform (-1,1) */
        return CMPLXF(2.0f * t1 - 1.0f, 2.0f * t2 - 1.0f);
    } else if (idist == 3) {
        /* real and imaginary parts each normal (0,1) */
        f32 r = sqrtf(-2.0f * logf(t1));
        f32 theta = TWOPI * t2;
        return CMPLXF(r * cosf(theta), r * sinf(theta));
    } else if (idist == 4) {
        /* uniform distribution on the unit disc abs(z) <= 1 */
        f32 r = sqrtf(t1);
        f32 theta = TWOPI * t2;
        return CMPLXF(r * cosf(theta), r * sinf(theta));
    } else if (idist == 5) {
        /* uniform distribution on the unit circle abs(z) = 1 */
        f32 theta = TWOPI * t2;
        return CMPLXF(cosf(theta), sinf(theta));
    }
    return CMPLXF(0.0f, 0.0f);
}

/**
 * clarnv_rng returns a vector of n random complex numbers from a uniform
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
void clarnv_rng(const INT idist, const INT n, c64* x,
                uint64_t state[static 4])
{
    if (n <= 0) {
        return;
    }

    for (INT i = 0; i < n; i++) {
        x[i] = clarnd_rng(idist, state);
    }
}
