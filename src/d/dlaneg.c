/**
 * @file dlaneg.c
 * @brief DLANEG computes the Sturm count.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"

/**
 * DLANEG computes the Sturm count, the number of negative pivots
 * encountered while factoring tridiagonal T - sigma I = L D L^T.
 * This implementation works directly on the factors without forming
 * the tridiagonal matrix T. The Sturm count is also the number of
 * eigenvalues of T less than sigma.
 *
 * This routine is called from DLARRB.
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
INT dlaneg(const INT n, const f64* D, const f64* lld,
           const f64 sigma, const f64 pivmin, const INT r)
{
    (void)pivmin;  /* Not used; requires IEEE-754 propagation */
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const INT BLKLEN = 128;

    INT negcnt = 0;

    /* I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
     * Fortran loop: DO BJ = 1, R-1 with inner loop J = BJ to MIN(BJ+BLKLEN-1, R-1)
     * In 0-based C: bj goes from 0 to r-1 (when r>0), using indices 0..r-1. */
    f64 t = -sigma;
    for (INT bj = 0; bj < r; bj += BLKLEN) {
        INT neg1 = 0;
        f64 bsav = t;
        INT jend = bj + BLKLEN - 1;
        if (jend > r - 1) jend = r - 1;
        for (INT j = bj; j <= jend; j++) {
            f64 dplus = D[j] + t;
            if (dplus < ZERO) neg1++;
            f64 tmp = t / dplus;
            t = tmp * lld[j] - sigma;
        }
        INT sawnan = disnan(t);
        /* Run a slower version of the above loop if a NaN is detected.
           A NaN should occur only with a zero pivot after an infinite
           pivot.  In that case, substituting 1 for T/DPLUS is the
           correct limit. */
        if (sawnan) {
            neg1 = 0;
            t = bsav;
            /* Recompute jend for the NaN-safe loop */
            jend = bj + BLKLEN - 1;
            if (jend > r - 1) jend = r - 1;
            for (INT j = bj; j <= jend; j++) {
                f64 dplus = D[j] + t;
                if (dplus < ZERO) neg1++;
                f64 tmp = t / dplus;
                if (disnan(tmp)) tmp = ONE;
                t = tmp * lld[j] - sigma;
            }
        }
        negcnt += neg1;
    }

    /* II) lower part: L D L^T - SIGMA I = U- D- U-^T
     * Fortran loop: DO BJ = N-1, R, -BLKLEN with inner loop J = BJ down to MAX(BJ-BLKLEN+1, R)
     * In 0-based C: bj goes from n-2 down to r (inclusive), using indices r..n-2. */
    f64 p = D[n - 1] - sigma;
    for (INT bj = n - 2; bj >= r; bj -= BLKLEN) {
        INT neg2 = 0;
        f64 bsav = p;
        INT jend = bj - BLKLEN + 1;
        if (jend < r) jend = r;
        for (INT j = bj; j >= jend; j--) {
            f64 dminus = lld[j] + p;
            if (dminus < ZERO) neg2++;
            f64 tmp = p / dminus;
            p = tmp * D[j] - sigma;
        }
        INT sawnan = disnan(p);
        /* As above, run a slower version that substitutes 1 for Inf/Inf. */
        if (sawnan) {
            neg2 = 0;
            p = bsav;
            for (INT j = bj; j >= jend; j--) {
                f64 dminus = lld[j] + p;
                if (dminus < ZERO) neg2++;
                f64 tmp = p / dminus;
                if (disnan(tmp)) tmp = ONE;
                p = tmp * D[j] - sigma;
            }
        }
        negcnt += neg2;
    }

    /* III) Twist index
       T was shifted by SIGMA initially. */
    f64 gamma = (t + sigma) + p;
    if (gamma < ZERO) negcnt++;

    return negcnt;
}
