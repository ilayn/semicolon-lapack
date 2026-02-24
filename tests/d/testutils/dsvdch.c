/**
 * @file dsvdch.c
 * @brief DSVDCH checks computed singular values of a bidiagonal matrix.
 */

#include "verify.h"
#include <math.h>

/**
 * DSVDCH checks to see if SVD(1) ,..., SVD(N) are accurate singular
 * values of the bidiagonal matrix B with diagonal entries
 * S(1) ,..., S(N) and superdiagonal entries E(1) ,..., E(N-1)).
 * It does this by expanding each SVD(I) into an interval
 * [SVD(I) * (1-EPS) , SVD(I) * (1+EPS)], merging overlapping intervals
 * if any, and using Sturm sequences to count and verify whether each
 * resulting interval has the correct number of singular values (using
 * DSVDCT). Here EPS=TOL*MAX(N/10,1)*MAZHEP, where MACHEP is the
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
void dsvdch(const INT n, const f64* s, const f64* e,
            const f64* svd, const f64 tol, INT* info)
{
    const f64 ONE = 1.0;

    *info = 0;
    if (n <= 0)
        return;

    f64 unfl = dlamch("Safe minimum");
    f64 ovfl = dlamch("Overflow");
    f64 eps = dlamch("Epsilon") * dlamch("Base");

    /* UNFLEP is chosen so that when an eigenvalue is multiplied by the
       scale factor sqrt(OVFL)*sqrt(sqrt(UNFL))/MX in DSVDCT, it exceeds
       sqrt(UNFL), which is the lower limit for DSVDCT. */

    f64 unflep = (sqrt(sqrt(unfl)) / sqrt(ovfl)) * svd[0] + unfl / eps;

    /* The value of EPS works best when TOL >= 10. */

    INT nover10 = n / 10;
    if (nover10 < 1) nover10 = 1;
    eps = tol * nover10 * eps;

    /* TPNT points to singular value at right endpoint of interval
       BPNT points to singular value at left  endpoint of interval */

    /* Fortran 1-based tpnt/bpnt → C 0-based */
    INT tpnt = 0;
    INT bpnt = 0;

    /* Begin loop over all intervals */

    while (tpnt < n) {
        f64 upper = (ONE + eps) * svd[tpnt] + unflep;
        f64 lower = (ONE - eps) * svd[bpnt] - unflep;
        if (lower <= unflep)
            lower = -upper;

        /* Begin loop merging overlapping intervals */

        while (bpnt < n - 1) {
            f64 tuppr = (ONE + eps) * svd[bpnt + 1] + unflep;
            if (tuppr < lower)
                break;

            /* Merge */

            bpnt++;
            lower = (ONE - eps) * svd[bpnt] - unflep;
            if (lower <= unflep)
                lower = -upper;
        }

        /* Count singular values in interval [ LOWER, UPPER ] */

        INT numl, numu;
        dsvdct(n, s, e, lower, &numl);
        dsvdct(n, s, e, upper, &numu);
        INT count = numu - numl;
        if (lower < 0.0)
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
