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
int slaneg(const int n, const float* D, const float* lld,
           const float sigma, const float pivmin, const int r)
{
    (void)pivmin;  /* Not used; requires IEEE-754 propagation */
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const int BLKLEN = 128;

    int negcnt = 0;

    /* I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
     * Fortran loop: DO BJ = 1, R-1 with inner loop J = BJ to MIN(BJ+BLKLEN-1, R-1)
     * In 0-based C: bj goes from 0 to r-1 (when r>0), using indices 0..r-1. */
    float t = -sigma;
    for (int bj = 0; bj < r; bj += BLKLEN) {
        int neg1 = 0;
        float bsav = t;
        int jend = bj + BLKLEN - 1;
        if (jend > r - 1) jend = r - 1;
        for (int j = bj; j <= jend; j++) {
            float dplus = D[j] + t;
            if (dplus < ZERO) neg1++;
            float tmp = t / dplus;
            t = tmp * lld[j] - sigma;
        }
        int sawnan = sisnan(t);
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
            for (int j = bj; j <= jend; j++) {
                float dplus = D[j] + t;
                if (dplus < ZERO) neg1++;
                float tmp = t / dplus;
                if (sisnan(tmp)) tmp = ONE;
                t = tmp * lld[j] - sigma;
            }
        }
        negcnt += neg1;
    }

    /* II) lower part: L D L^T - SIGMA I = U- D- U-^T
     * Fortran loop: DO BJ = N-1, R, -BLKLEN with inner loop J = BJ down to MAX(BJ-BLKLEN+1, R)
     * In 0-based C: bj goes from n-2 down to r (inclusive), using indices r..n-2. */
    float p = D[n - 1] - sigma;
    for (int bj = n - 2; bj >= r; bj -= BLKLEN) {
        int neg2 = 0;
        float bsav = p;
        int jend = bj - BLKLEN + 1;
        if (jend < r) jend = r;
        for (int j = bj; j >= jend; j--) {
            float dminus = lld[j] + p;
            if (dminus < ZERO) neg2++;
            float tmp = p / dminus;
            p = tmp * D[j] - sigma;
        }
        int sawnan = sisnan(p);
        /* As above, run a slower version that substitutes 1 for Inf/Inf. */
        if (sawnan) {
            neg2 = 0;
            p = bsav;
            for (int j = bj; j >= jend; j--) {
                float dminus = lld[j] + p;
                if (dminus < ZERO) neg2++;
                float tmp = p / dminus;
                if (sisnan(tmp)) tmp = ONE;
                p = tmp * D[j] - sigma;
            }
        }
        negcnt += neg2;
    }

    /* III) Twist index
       T was shifted by SIGMA initially. */
    float gamma = (t + sigma) + p;
    if (gamma < ZERO) negcnt++;

    return negcnt;
}
