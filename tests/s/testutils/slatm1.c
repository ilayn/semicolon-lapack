/**
 * @file slatm1.c
 * @brief SLATM1 computes the entries of D(1..N) as specified by MODE, COND
 *        and IRSIGN.
 *
 * Faithful port of LAPACK TESTING/MATGEN/slatm1.f
 * Uses xoshiro256+ RNG instead of LAPACK's 48-bit LCG.
 */

#include <math.h>
#include "verify.h"
#include "test_rng.h"


/**
 * SLATM1 computes the entries of D(1..N) as specified by
 * MODE, COND and IRSIGN.
 *
 * @param[in] mode
 *     On entry describes how D is to be computed:
 *     MODE = 0 means do not change D.
 *     MODE = 1 sets D(0)=1 and D(1:N-1)=1.0/COND
 *     MODE = 2 sets D(0:N-2)=1 and D(N-1)=1.0/COND
 *     MODE = 3 sets D(I)=COND**(-(I)/(N-1))
 *     MODE = 4 sets D(i)=1 - (i)/(N-1)*(1 - 1/COND)
 *     MODE = 5 sets D to random numbers in the range
 *              ( 1/COND , 1 ) such that their logarithms
 *              are uniformly distributed.
 *     MODE = 6 set D to random numbers from same distribution
 *              as the rest of the matrix.
 *     MODE < 0 has the same meaning as ABS(MODE), except that
 *        the order of the elements of D is reversed.
 *     Thus if MODE is positive, D has entries ranging from
 *        1 to 1/COND, if negative, from 1/COND to 1,
 *
 * @param[in] cond
 *     On entry, used as described under MODE above.
 *     If used, it must be >= 1.
 *
 * @param[in] irsign
 *     On entry, if MODE neither -6, 0 nor 6, determines sign of
 *     entries of D:
 *     0 => leave entries of D unchanged
 *     1 => multiply each entry of D by 1 or -1 with probability .5
 *
 * @param[in] idist
 *     On entry, IDIST specifies the type of distribution to be
 *     used to generate a random matrix:
 *     1 => UNIFORM( 0, 1 )
 *     2 => UNIFORM( -1, 1 )
 *     3 => NORMAL( 0, 1 )
 *
 * @param[out] d
 *     Array to be computed according to MODE, COND and IRSIGN.
 *     Dimension (n). 0-based indexing.
 *
 * @param[in] n
 *     Number of entries of D.
 *
 * @param[out] info
 *     0  => normal termination
 *    -1  => if MODE not in range -6 to 6
 *    -2  => if MODE neither -6, 0 nor 6, and IRSIGN neither 0 nor 1
 *    -3  => if MODE neither -6, 0 nor 6 and COND less than 1
 *    -4  => if MODE equals 6 or -6 and IDIST not in range 1 to 3
 *    -7  => if N negative
 */
void slatm1(
    const INT mode,
    const f32 cond,
    const INT irsign,
    const INT idist,
    f32* d,
    const INT n,
    INT* info,
    uint64_t state[static 4])
{
    const f32 ONE = 1.0f;
    const f32 HALF = 0.5f;

    INT i;
    f32 alpha, temp;
    INT absmode;

    *info = 0;

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /* Set INFO if an error */
    if (mode < -6 || mode > 6) {
        *info = -1;
    } else if ((mode != -6 && mode != 0 && mode != 6) &&
               (irsign != 0 && irsign != 1)) {
        *info = -2;
    } else if ((mode != -6 && mode != 0 && mode != 6) && cond < ONE) {
        *info = -3;
    } else if ((mode == 6 || mode == -6) && (idist < 1 || idist > 3)) {
        *info = -4;
    } else if (n < 0) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("SLATM1", -(*info));
        return;
    }

    /* Compute D according to COND and MODE */
    if (mode != 0) {
        absmode = (mode < 0) ? -mode : mode;

        switch (absmode) {
            case 1:
                /* One large D value: D[0]=1, D[1:n-1]=1/cond */
                for (i = 0; i < n; i++) {
                    d[i] = ONE / cond;
                }
                d[0] = ONE;
                break;

            case 2:
                /* One small D value: D[0:n-2]=1, D[n-1]=1/cond */
                for (i = 0; i < n; i++) {
                    d[i] = ONE;
                }
                d[n - 1] = ONE / cond;
                break;

            case 3:
                /* Exponentially distributed D values: D[i]=cond^(-i/(n-1)) */
                d[0] = ONE;
                if (n > 1) {
                    alpha = powf(cond, -ONE / (f32)(n - 1));
                    for (i = 1; i < n; i++) {
                        d[i] = powf(alpha, (f32)i);
                    }
                }
                break;

            case 4:
                /* Arithmetically distributed D values:
                 * D[i] = 1 - i/(n-1)*(1 - 1/cond) */
                d[0] = ONE;
                if (n > 1) {
                    temp = ONE / cond;
                    alpha = (ONE - temp) / (f32)(n - 1);
                    for (i = 1; i < n; i++) {
                        d[i] = (f32)(n - 1 - i) * alpha + temp;
                    }
                }
                break;

            case 5:
                /* Randomly distributed D values on (1/cond, 1): */
                alpha = logf(ONE / cond);
                for (i = 0; i < n; i++) {
                    d[i] = expf(alpha * rng_uniform_f32(state));
                }
                break;

            case 6:
                /* Randomly distributed D values from IDIST */
                switch (idist) {
                    case 1:  /* UNIFORM(0, 1) */
                        for (i = 0; i < n; i++) {
                            d[i] = rng_uniform_f32(state);
                        }
                        break;
                    case 2:  /* UNIFORM(-1, 1) */
                        for (i = 0; i < n; i++) {
                            d[i] = 2.0f * rng_uniform_f32(state) - 1.0f;
                        }
                        break;
                    case 3:  /* NORMAL(0, 1) */
                        for (i = 0; i < n; i++) {
                            d[i] = rng_normal_f32(state);
                        }
                        break;
                }
                break;
        }

        /* If MODE neither -6 nor 0 nor 6, and IRSIGN = 1, assign
         * random signs to D */
        if ((mode != -6 && mode != 0 && mode != 6) && irsign == 1) {
            for (i = 0; i < n; i++) {
                temp = rng_uniform_f32(state);
                if (temp > HALF) {
                    d[i] = -d[i];
                }
            }
        }

        /* Reverse if MODE < 0 */
        if (mode < 0) {
            for (i = 0; i < n / 2; i++) {
                temp = d[i];
                d[i] = d[n - 1 - i];
                d[n - 1 - i] = temp;
            }
        }
    }
}
