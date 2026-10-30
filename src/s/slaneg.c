/**
 * @file slaneg.c
 * @brief SLANEG computes the Sturm count.
 */

#include "semicolon_lapack_single.h"

/**
 * SLANEG computes the Sturm count, the number of negative pivots
 * encountered while factoring tridiagonal T - sigma I = L D L^T.
 * This implementation works directly on the factors without forming
 * the tridiagonal matrix T. The Sturm count is also the number of
 * eigenvalues of T less than sigma.
 *
 * This routine is called from SLARRB.
 *
 * The current routine does not use the PIVMIN parameter but rather
 * requires IEEE-754 propagation of Infinities and NaNs.
 *
 * @param[in]     n       The order of the matrix.
 * @param[in]     D       Double precision array, dimension (n).
 *                        The n diagonal elements of the diagonal matrix D.
 * @param[in]     lld     Double precision array, dimension (n-1).
 *                        The (n-1) elements L(i)*L(i)*D(i).
 * @param[in]     sigma   Shift amount in T - sigma I = L D L^T.
 * @param[in]     pivmin  The minimum pivot in the Sturm sequence.
 * @param[in]     r       The twist index for the twisted factorization.
 *
 * @return The number of negative pivots (Sturm count).
 */
INT slaneg(const INT n, const f32* D, const f32* lld,
           const f32 sigma, const f32 pivmin, const INT r)
{
    (void)pivmin;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const INT BLKLEN = 128;

    INT negcnt = 0;

    /* I) upper part: L D L^T - SIGMA I = L+ D+ L+^T */
    f32 t = -sigma;
    for (INT bj = 0; bj < r; bj += BLKLEN) {
        INT neg1 = 0;
        f32 bsav = t;
        INT jend = bj + BLKLEN - 1;
        if (jend > r - 1) jend = r - 1;
        for (INT j = bj; j <= jend; j++) {
            f32 dplus = D[j] + t;
            if (dplus < ZERO) neg1++;
            f32 tmp = t / dplus;
            t = tmp * lld[j] - sigma;
        }
        INT sawnan = sisnan(t);
        /* Run a slower version of the above loop if a NaN is detected.
           A NaN should occur only with a zero pivot after an infinite
           pivot.  In that case, substituting 1 for T/DPLUS is the
           correct limit. */
        if (sawnan) {
            neg1 = 0;
            t = bsav;
            jend = bj + BLKLEN - 1;
            if (jend > r - 1) jend = r - 1;
            for (INT j = bj; j <= jend; j++) {
                f32 dplus = D[j] + t;
                if (dplus < ZERO) neg1++;
                f32 tmp = t / dplus;
                if (sisnan(tmp)) tmp = ONE;
                t = tmp * lld[j] - sigma;
            }
        }
        negcnt += neg1;
    }

    /* II) lower part: L D L^T - SIGMA I = U- D- U-^T */
    f32 p = D[n - 1] - sigma;
    for (INT bj = n - 2; bj >= r; bj -= BLKLEN) {
        INT neg2 = 0;
        f32 bsav = p;
        INT jend = bj - BLKLEN + 1;
        if (jend < r) jend = r;
        for (INT j = bj; j >= jend; j--) {
            f32 dminus = lld[j] + p;
            if (dminus < ZERO) neg2++;
            f32 tmp = p / dminus;
            p = tmp * D[j] - sigma;
        }
        INT sawnan = sisnan(p);
        /* As above, run a slower version that substitutes 1 for Inf/Inf. */
        if (sawnan) {
            neg2 = 0;
            p = bsav;
            for (INT j = bj; j >= jend; j--) {
                f32 dminus = lld[j] + p;
                if (dminus < ZERO) neg2++;
                f32 tmp = p / dminus;
                if (sisnan(tmp)) tmp = ONE;
                p = tmp * D[j] - sigma;
            }
        }
        negcnt += neg2;
    }

    /* III) Twist index
       T was shifted by SIGMA initially. */
    f32 gamma = (t + sigma) + p;
    if (gamma < ZERO) negcnt++;

    return negcnt;
}
