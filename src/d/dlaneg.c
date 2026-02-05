/**
 * @file dlaneg.c
 * @brief DLANEG computes the Sturm count.
 */

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
int dlaneg(const int n, const double* D, const double* lld,
           const double sigma, const double pivmin, const int r)
{
    (void)pivmin;  /* Not used; requires IEEE-754 propagation */
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const int BLKLEN = 128;

    int negcnt = 0;

    /* I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
     * Fortran loop: DO BJ = 1, R-1 with inner loop J = BJ to MIN(BJ+BLKLEN-1, R-1)
     * In 0-based C: bj goes from 0 to r-1 (when r>0), using indices 0..r-1. */
    double t = -sigma;
    for (int bj = 0; bj < r; bj += BLKLEN) {
        int neg1 = 0;
        double bsav = t;
        int jend = bj + BLKLEN - 1;
        if (jend > r - 1) jend = r - 1;
        for (int j = bj; j <= jend; j++) {
            double dplus = D[j] + t;
            if (dplus < ZERO) neg1++;
            double tmp = t / dplus;
            t = tmp * lld[j] - sigma;
        }
        int sawnan = disnan(t);
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
                double dplus = D[j] + t;
                if (dplus < ZERO) neg1++;
                double tmp = t / dplus;
                if (disnan(tmp)) tmp = ONE;
                t = tmp * lld[j] - sigma;
            }
        }
        negcnt += neg1;
    }

    /* II) lower part: L D L^T - SIGMA I = U- D- U-^T
     * Fortran loop: DO BJ = N-1, R, -BLKLEN with inner loop J = BJ down to MAX(BJ-BLKLEN+1, R)
     * In 0-based C: bj goes from n-2 down to r (inclusive), using indices r..n-2. */
    double p = D[n - 1] - sigma;
    for (int bj = n - 2; bj >= r; bj -= BLKLEN) {
        int neg2 = 0;
        double bsav = p;
        int jend = bj - BLKLEN + 1;
        if (jend < r) jend = r;
        for (int j = bj; j >= jend; j--) {
            double dminus = lld[j] + p;
            if (dminus < ZERO) neg2++;
            double tmp = p / dminus;
            p = tmp * D[j] - sigma;
        }
        int sawnan = disnan(p);
        /* As above, run a slower version that substitutes 1 for Inf/Inf. */
        if (sawnan) {
            neg2 = 0;
            p = bsav;
            for (int j = bj; j >= jend; j--) {
                double dminus = lld[j] + p;
                if (dminus < ZERO) neg2++;
                double tmp = p / dminus;
                if (disnan(tmp)) tmp = ONE;
                p = tmp * D[j] - sigma;
            }
        }
        negcnt += neg2;
    }

    /* III) Twist index
       T was shifted by SIGMA initially. */
    double gamma = (t + sigma) + p;
    if (gamma < ZERO) negcnt++;

    return negcnt;
}
