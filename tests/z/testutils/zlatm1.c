/**
 * @file zlatm1.c
 * @brief ZLATM1 computes the entries of D(1..N) as specified by MODE, COND
 *        and IRSIGN.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlatm1.f
 * Uses xoshiro256+ RNG instead of LAPACK's 48-bit LCG.
 */

#include <math.h>
#include "verify.h"
#include "test_rng.h"

/**
 * ZLATM1 computes the entries of D(1..N) as specified by
 * MODE, COND and IRSIGN. IDIST and state determine the generation
 * of random numbers. ZLATM1 is called by ZLATMR to generate
 * random test matrices for LAPACK programs.
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
 *     1 => multiply each entry of D by random complex number
 *          uniformly distributed with absolute value 1
 *
 * @param[in] idist
 *     On entry, IDIST specifies the type of distribution to be
 *     used to generate a random matrix:
 *     1 => real and imaginary parts each UNIFORM( 0, 1 )
 *     2 => real and imaginary parts each UNIFORM( -1, 1 )
 *     3 => real and imaginary parts each NORMAL( 0, 1 )
 *     4 => complex number uniform in DISK( 0, 1 )
 *
 * @param[out] d
 *     Complex array to be computed according to MODE, COND and IRSIGN.
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
 *    -4  => if MODE equals 6 or -6 and IDIST not in range 1 to 4
 *    -7  => if N negative
 *
 * @param[in,out] state
 *     RNG state array (xoshiro256+), already initialized by the caller.
 */
void zlatm1(
    const INT mode,
    const f64 cond,
    const INT irsign,
    const INT idist,
    c128* d,
    const INT n,
    INT* info,
    uint64_t state[static 4])
{
    const f64 ONE = 1.0;

    INT i;
    f64 alpha, temp;
    c128 ctemp;
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
    } else if ((mode == 6 || mode == -6) && (idist < 1 || idist > 4)) {
        *info = -4;
    } else if (n < 0) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("ZLATM1", -(*info));
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
                    alpha = pow(cond, -ONE / (f64)(n - 1));
                    for (i = 1; i < n; i++) {
                        d[i] = pow(alpha, (f64)i);
                    }
                }
                break;

            case 4:
                /* Arithmetically distributed D values:
                 * D[i] = 1 - i/(n-1)*(1 - 1/cond) */
                d[0] = ONE;
                if (n > 1) {
                    temp = ONE / cond;
                    alpha = (ONE - temp) / (f64)(n - 1);
                    for (i = 1; i < n; i++) {
                        d[i] = (f64)(n - 1 - i) * alpha + temp;
                    }
                }
                break;

            case 5:
                /* Randomly distributed D values on (1/cond, 1): */
                alpha = log(ONE / cond);
                for (i = 0; i < n; i++) {
                    d[i] = exp(alpha * rng_uniform(state));
                }
                break;

            case 6:
                /* Randomly distributed D values from IDIST */
                zlarnv_rng(idist, n, d, state);
                break;
        }

        /* If MODE neither -6 nor 0 nor 6, and IRSIGN = 1, multiply
         * each entry of D by random complex number with absolute value 1 */
        if ((mode != -6 && mode != 0 && mode != 6) && irsign == 1) {
            for (i = 0; i < n; i++) {
                ctemp = zlarnd_rng(3, state);
                d[i] = d[i] * (ctemp / cabs(ctemp));
            }
        }

        /* Reverse if MODE < 0 */
        if (mode < 0) {
            for (i = 0; i < n / 2; i++) {
                ctemp = d[i];
                d[i] = d[n - 1 - i];
                d[n - 1 - i] = ctemp;
            }
        }
    }
}
