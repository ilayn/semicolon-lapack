/** @file slar1v.c
 * @brief SLAR1V computes the (scaled) r-th column of the inverse of the
 *        submatrix in rows b1 through bn of the tridiagonal matrix
 *        L D L^T - lambda I.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAR1V computes the (scaled) r-th column of the inverse of
 * the submatrix in rows B1 through BN of the tridiagonal matrix
 * L D L^T - sigma I. When sigma is close to an eigenvalue, the
 * computed vector is an accurate eigenvector. Usually, r corresponds
 * to the index where the eigenvector is largest in magnitude.
 * The following steps accomplish this computation:
 * (a) Stationary qd transform,  L D L^T - sigma I = L(+) D(+) L(+)^T,
 * (b) Progressive qd transform, L D L^T - sigma I = U(-) D(-) U(-)^T,
 * (c) Computation of the diagonal elements of the inverse of
 *     L D L^T - sigma I by combining the above transforms, and choosing
 *     r as the index where the diagonal of the inverse is (one of the)
 *     largest in magnitude.
 * (d) Computation of the (scaled) r-th column of the inverse using the
 *     twisted factorization obtained by combining the top part of the
 *     the stationary and the bottom part of the progressive transform.
 *
 * @param[in]     n       The order of the matrix L D L^T. n >= 0.
 * @param[in]     b1      First index of the submatrix (0-based).
 * @param[in]     bn      Last index of the submatrix (0-based).
 * @param[in]     lambda  The shift. Should be a good approximation to an
 *                        eigenvalue of L D L^T.
 * @param[in]     D       Double precision array, dimension (n).
 *                        The n diagonal elements of the diagonal matrix D.
 * @param[in]     L       Double precision array, dimension (n-1).
 *                        The (n-1) subdiagonal elements of the unit
 *                        bidiagonal matrix L.
 * @param[in]     LD      Double precision array, dimension (n-1).
 *                        The n-1 elements L(i)*D(i).
 * @param[in]     LLD     Double precision array, dimension (n-1).
 *                        The n-1 elements L(i)*L(i)*D(i).
 * @param[in]     pivmin  The minimum pivot in the Sturm sequence.
 * @param[in]     gaptol  Tolerance that indicates when eigenvector entries
 *                        are negligible w.r.t. their contribution to the
 *                        residual.
 * @param[in,out] Z       Double precision array, dimension (n).
 *                        On input, all entries of Z must be set to 0.
 *                        On output, Z contains the (scaled) r-th column of
 *                        the inverse. The scaling is such that Z[r] equals 1.
 * @param[in]     wantnc  If nonzero, NEGCNT will be computed.
 * @param[out]    negcnt  If wantnc is nonzero, the number of pivots < pivmin
 *                        in the matrix factorization L D L^T; otherwise -1.
 * @param[out]    ztz     The square of the 2-norm of Z.
 * @param[out]    mingma  The reciprocal of the largest (in magnitude) diagonal
 *                        element of the inverse of L D L^T - sigma I.
 * @param[in,out] r       The twist index (0-based).
 *                        On input, if r < 0, r is set to the index where
 *                        (L D L^T - sigma I)^{-1} is largest in magnitude.
 *                        If 0 <= r < n, r is unchanged.
 *                        On output, contains the twist index used to compute Z.
 * @param[out]    isuppz  Integer array, dimension (2).
 *                        The support of the vector Z (0-based indices).
 *                        Z is nonzero only in elements isuppz[0] through
 *                        isuppz[1].
 * @param[out]    nrminv  1/sqrt(ztz).
 * @param[out]    resid   The residual of the FP vector: |mingma|/sqrt(ztz).
 * @param[out]    rqcorr  The Rayleigh Quotient correction to lambda.
 * @param[out]    work    Double precision array, dimension (4*n).
 */
void slar1v(const int n, const int b1, const int bn, const f32 lambda,
            const f32* restrict D, const f32* restrict L,
            const f32* restrict LD, const f32* restrict LLD,
            const f32 pivmin, const f32 gaptol,
            f32* restrict Z, const int wantnc, int* negcnt,
            f32* ztz, f32* mingma, int* r,
            int* restrict isuppz, f32* nrminv, f32* resid,
            f32* rqcorr, f32* restrict work)
{
    /* Local variables */
    int i, r1, r2, neg1, neg2;
    int sawnan1, sawnan2;
    f32 dplus, dminus, s, tmp, eps;

    /* Workspace segment offsets (0-based):
     *   work[indlpl .. indlpl+n-2]: L+ values
     *   work[indumn .. indumn+n-2]: U- values
     *   work[inds   .. inds+n-1]:   S  values (stationary transform)
     *   work[indp   .. indp+n-1]:   P  values (progressive transform) */
    const int indlpl = 0;
    const int indumn = n;
    const int inds   = 2 * n;
    const int indp   = 3 * n;

    eps = slamch("Precision");

    /* Determine the range [r1, r2] for the twist index search.
     * r < 0 means "search for best twist index". */
    if (*r < 0) {
        r1 = b1;
        r2 = bn;
    } else {
        r1 = *r;
        r2 = *r;
    }

    /* Initialize the S segment at position b1 */
    if (b1 == 0) {
        work[inds + b1] = 0.0f;
    } else {
        work[inds + b1] = LLD[b1 - 1];
    }

    /*
     * Compute the stationary transform (using the differential form)
     * until the index r2.
     */
    neg1 = 0;
    s = work[inds + b1] - lambda;

    /* Forward loop from b1 to r1-1 */
    for (i = b1; i < r1; i++) {
        dplus = D[i] + s;
        work[indlpl + i] = LD[i] / dplus;
        if (dplus < 0.0f) {
            neg1 = neg1 + 1;
        }
        work[inds + i + 1] = s * work[indlpl + i] * L[i];
        s = work[inds + i + 1] - lambda;
    }
    sawnan1 = sisnan(s);

    if (!sawnan1) {
        /* Forward loop from r1 to r2-1 */
        for (i = r1; i < r2; i++) {
            dplus = D[i] + s;
            work[indlpl + i] = LD[i] / dplus;
            work[inds + i + 1] = s * work[indlpl + i] * L[i];
            s = work[inds + i + 1] - lambda;
        }
        sawnan1 = sisnan(s);
    }

    if (sawnan1) {
        /* Slower NaN-safe version of the stationary transform */
        neg1 = 0;
        s = work[inds + b1] - lambda;

        for (i = b1; i < r1; i++) {
            dplus = D[i] + s;
            if (fabsf(dplus) < pivmin) {
                dplus = -pivmin;
            }
            work[indlpl + i] = LD[i] / dplus;
            if (dplus < 0.0f) {
                neg1 = neg1 + 1;
            }
            work[inds + i + 1] = s * work[indlpl + i] * L[i];
            if (work[indlpl + i] == 0.0f) {
                work[inds + i + 1] = LLD[i];
            }
            s = work[inds + i + 1] - lambda;
        }

        for (i = r1; i < r2; i++) {
            dplus = D[i] + s;
            if (fabsf(dplus) < pivmin) {
                dplus = -pivmin;
            }
            work[indlpl + i] = LD[i] / dplus;
            work[inds + i + 1] = s * work[indlpl + i] * L[i];
            if (work[indlpl + i] == 0.0f) {
                work[inds + i + 1] = LLD[i];
            }
            s = work[inds + i + 1] - lambda;
        }
    }

    /*
     * Compute the progressive transform (using the differential form)
     * until the index r1.
     */
    neg2 = 0;
    work[indp + bn] = D[bn] - lambda;

    /* Backward loop from bn-1 down to r1 */
    for (i = bn - 1; i >= r1; i--) {
        dminus = LLD[i] + work[indp + i + 1];
        tmp = D[i] / dminus;
        if (dminus < 0.0f) {
            neg2 = neg2 + 1;
        }
        work[indumn + i] = L[i] * tmp;
        work[indp + i] = work[indp + i + 1] * tmp - lambda;
    }
    tmp = work[indp + r1];
    sawnan2 = sisnan(tmp);

    if (sawnan2) {
        /* Slower NaN-safe version of the progressive transform */
        neg2 = 0;
        work[indp + bn] = D[bn] - lambda;

        for (i = bn - 1; i >= r1; i--) {
            dminus = LLD[i] + work[indp + i + 1];
            if (fabsf(dminus) < pivmin) {
                dminus = -pivmin;
            }
            tmp = D[i] / dminus;
            if (dminus < 0.0f) {
                neg2 = neg2 + 1;
            }
            work[indumn + i] = L[i] * tmp;
            work[indp + i] = work[indp + i + 1] * tmp - lambda;
            if (tmp == 0.0f) {
                work[indp + i] = D[i] - lambda;
            }
        }
    }


    /*
     * Find the index (from r1 to r2) of the largest (in magnitude)
     * diagonal element of the inverse.
     */
    *mingma = work[inds + r1] + work[indp + r1];
    if (*mingma < 0.0f) {
        neg1 = neg1 + 1;
    }
    if (wantnc) {
        *negcnt = neg1 + neg2;
    } else {
        *negcnt = -1;
    }
    if (fabsf(*mingma) == 0.0f) {
        *mingma = eps * work[inds + r1];
    }
    *r = r1;

    for (i = r1; i < r2; i++) {
        tmp = work[inds + i + 1] + work[indp + i + 1];
        if (tmp == 0.0f) {
            tmp = eps * work[inds + i + 1];
        }
        if (fabsf(tmp) <= fabsf(*mingma)) {
            *mingma = tmp;
            *r = i + 1;
        }
    }

    /*
     * Compute the FP vector: solve N^T v = e_r
     */
    isuppz[0] = b1;
    isuppz[1] = bn;
    Z[*r] = 1.0f;
    *ztz = 1.0f;

    /*
     * Compute the FP vector upwards from r
     */
    if (!sawnan1 && !sawnan2) {
        for (i = *r - 1; i >= b1; i--) {
            Z[i] = -(work[indlpl + i] * Z[i + 1]);
            if ((fabsf(Z[i]) + fabsf(Z[i + 1])) * fabsf(LD[i]) < gaptol) {
                Z[i] = 0.0f;
                isuppz[0] = i + 1;
                break;
            }
            *ztz = *ztz + Z[i] * Z[i];
        }
    } else {
        /* Slower NaN-safe upward loop */
        for (i = *r - 1; i >= b1; i--) {
            if (Z[i + 1] == 0.0f) {
                Z[i] = -(LD[i + 1] / LD[i]) * Z[i + 2];
            } else {
                Z[i] = -(work[indlpl + i] * Z[i + 1]);
            }
            if ((fabsf(Z[i]) + fabsf(Z[i + 1])) * fabsf(LD[i]) < gaptol) {
                Z[i] = 0.0f;
                isuppz[0] = i + 1;
                break;
            }
            *ztz = *ztz + Z[i] * Z[i];
        }
    }

    /*
     * Compute the FP vector downwards from r
     */
    if (!sawnan1 && !sawnan2) {
        for (i = *r; i < bn; i++) {
            Z[i + 1] = -(work[indumn + i] * Z[i]);
            if ((fabsf(Z[i]) + fabsf(Z[i + 1])) * fabsf(LD[i]) < gaptol) {
                Z[i + 1] = 0.0f;
                isuppz[1] = i;
                break;
            }
            *ztz = *ztz + Z[i + 1] * Z[i + 1];
        }
    } else {
        /* Slower NaN-safe downward loop */
        for (i = *r; i < bn; i++) {
            if (Z[i] == 0.0f) {
                Z[i + 1] = -(LD[i - 1] / LD[i]) * Z[i - 1];
            } else {
                Z[i + 1] = -(work[indumn + i] * Z[i]);
            }
            if ((fabsf(Z[i]) + fabsf(Z[i + 1])) * fabsf(LD[i]) < gaptol) {
                Z[i + 1] = 0.0f;
                isuppz[1] = i;
                break;
            }
            *ztz = *ztz + Z[i + 1] * Z[i + 1];
        }
    }

    /*
     * Compute quantities for convergence test
     */
    tmp = 1.0f / *ztz;
    *nrminv = sqrtf(tmp);
    *resid = fabsf(*mingma) * (*nrminv);
    *rqcorr = (*mingma) * tmp;

}
