/**
 * @file ssvdch.c
 * @brief SSVDCH checks computed singular values of a bidiagonal matrix.
 */

#include "verify.h"
#include <math.h>

extern f32 slamch(const char* cmach);

/**
 * SSVDCH checks to see if SVD(1) ,..., SVD(N) are accurate singular
 * values of the bidiagonal matrix B with diagonal entries
 * S(1) ,..., S(N) and superdiagonal entries E(1) ,..., E(N-1)).
 * It does this by expanding each SVD(I) into an interval
 * [SVD(I) * (1-EPS) , SVD(I) * (1+EPS)], merging overlapping intervals
 * if any, and using Sturm sequences to count and verify whether each
 * resulting interval has the correct number of singular values (using
 * SSVDCT). Here EPS=TOL*MAX(N/10,1)*MAZHEP, where MACHEP is the
 * machine precision. The routine assumes the singular values are sorted
 * with SVD(1) the largest and SVD(N) smallest.  If each interval
 * contains the correct number of singular values, INFO = 0 is returned,
 * otherwise INFO is the index of the first singular value in the first
 * bad interval.
 *
 * @param[in]     n      The dimension of the bidiagonal matrix B.
 * @param[in]     s      Double precision array, dimension (n).
 *                       The diagonal entries of the bidiagonal matrix B.
 * @param[in]     e      Double precision array, dimension (n-1).
 *                       The superdiagonal entries of the bidiagonal matrix B.
 * @param[in]     svd    Double precision array, dimension (n).
 *                       The computed singular values to be checked.
 * @param[in]     tol    Error tolerance for checking, a multiplier of the
 *                       machine precision.
 * @param[out]    info   = 0 if the singular values are all correct (to within
 *                           1 +- TOL*MAZHEPS)
 *                       > 0 if the interval containing the INFO-th singular value
 *                           contains the incorrect number of singular values.
 */
void ssvdch(const int n, const f32* s, const f32* e,
            const f32* svd, const f32 tol, int* info)
{
    const f32 ONE = 1.0f;

    *info = 0;
    if (n <= 0)
        return;

    f32 unfl = slamch("Safe minimum");
    f32 ovfl = slamch("Overflow");
    f32 eps = slamch("Epsilon") * slamch("Base");

    /* UNFLEP is chosen so that when an eigenvalue is multiplied by the
       scale factor sqrt(OVFL)*sqrt(sqrt(UNFL))/MX in SSVDCT, it exceeds
       sqrt(UNFL), which is the lower limit for SSVDCT. */

    f32 unflep = (sqrtf(sqrtf(unfl)) / sqrtf(ovfl)) * svd[0] + unfl / eps;

    /* The value of EPS works best when TOL >= 10. */

    int nover10 = n / 10;
    if (nover10 < 1) nover10 = 1;
    eps = tol * nover10 * eps;

    /* TPNT points to singular value at right endpoint of interval
       BPNT points to singular value at left  endpoint of interval */

    /* Fortran 1-based tpnt/bpnt → C 0-based */
    int tpnt = 0;
    int bpnt = 0;

    /* Begin loop over all intervals */

    while (tpnt < n) {
        f32 upper = (ONE + eps) * svd[tpnt] + unflep;
        f32 lower = (ONE - eps) * svd[bpnt] - unflep;
        if (lower <= unflep)
            lower = -upper;

        /* Begin loop merging overlapping intervals */

        while (bpnt < n - 1) {
            f32 tuppr = (ONE + eps) * svd[bpnt + 1] + unflep;
            if (tuppr < lower)
                break;

            /* Merge */

            bpnt++;
            lower = (ONE - eps) * svd[bpnt] - unflep;
            if (lower <= unflep)
                lower = -upper;
        }

        /* Count singular values in interval [ LOWER, UPPER ] */

        int numl, numu;
        ssvdct(n, s, e, lower, &numl);
        ssvdct(n, s, e, upper, &numu);
        int count = numu - numl;
        if (lower < 0.0f)
            count = count / 2;
        if (count != bpnt - tpnt + 1) {

            /* Wrong number of singular values in interval */

            *info = tpnt + 1;
            return;
        }
        tpnt = bpnt + 1;
        bpnt = tpnt;
    }
}
