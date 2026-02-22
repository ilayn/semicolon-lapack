/**
 * @file dorbdb5.c
 * @brief DORBDB5 orthogonalizes a column vector with respect to the columns of a matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DORBDB5 orthogonalizes the column vector
 *      X = [ X1 ]
 *          [ X2 ]
 * with respect to the columns of
 *      Q = [ Q1 ] .
 *          [ Q2 ]
 * The columns of Q must be orthonormal.
 *
 * If the projection is zero according to Kahan's "twice is enough"
 * criterion, then some other vector from the orthogonal complement
 * is returned.
 *
 * @param[in] m1
 *          The dimension of X1 and the number of rows in Q1. 0 <= m1.
 *
 * @param[in] m2
 *          The dimension of X2 and the number of rows in Q2. 0 <= m2.
 *
 * @param[in] n
 *          The number of columns in Q1 and Q2. 0 <= n.
 *
 * @param[in,out] X1
 *          Double precision array, dimension (m1).
 *          On entry, the top part of the vector to be orthogonalized.
 *          On exit, the top part of the projected vector.
 *
 * @param[in] incx1
 *          Increment for entries of X1.
 *
 * @param[in,out] X2
 *          Double precision array, dimension (m2).
 *          On entry, the bottom part of the vector to be orthogonalized.
 *          On exit, the bottom part of the projected vector.
 *
 * @param[in] incx2
 *          Increment for entries of X2.
 *
 * @param[in] Q1
 *          Double precision array, dimension (ldq1, n).
 *          The top part of the orthonormal basis matrix.
 *
 * @param[in] ldq1
 *          The leading dimension of Q1. ldq1 >= m1.
 *
 * @param[in] Q2
 *          Double precision array, dimension (ldq2, n).
 *          The bottom part of the orthonormal basis matrix.
 *
 * @param[in] ldq2
 *          The leading dimension of Q2. ldq2 >= m2.
 *
 * @param[out] work
 *          Double precision array, dimension (lwork).
 *
 * @param[in] lwork
 *          The dimension of the array work. lwork >= n.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dorbdb5(
    const INT m1,
    const INT m2,
    const INT n,
    f64* restrict X1,
    const INT incx1,
    f64* restrict X2,
    const INT incx2,
    const f64* restrict Q1,
    const INT ldq1,
    const f64* restrict Q2,
    const INT ldq2,
    f64* restrict work,
    const INT lwork,
    INT* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    INT childinfo, i, j;
    f64 eps, norm, scl, ssq;

    *info = 0;
    if (m1 < 0) {
        *info = -1;
    } else if (m2 < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (incx1 < 1) {
        *info = -5;
    } else if (incx2 < 1) {
        *info = -7;
    } else if (ldq1 < (1 > m1 ? 1 : m1)) {
        *info = -9;
    } else if (ldq2 < (1 > m2 ? 1 : m2)) {
        *info = -11;
    } else if (lwork < n) {
        *info = -13;
    }

    if (*info != 0) {
        xerbla("DORBDB5", -(*info));
        return;
    }

    eps = dlamch("P");

    scl = zero;
    ssq = zero;
    dlassq(m1, X1, incx1, &scl, &ssq);
    dlassq(m2, X2, incx2, &scl, &ssq);
    norm = scl * sqrt(ssq);

    if (norm > n * eps) {
        cblas_dscal(m1, one / norm, X1, incx1);
        cblas_dscal(m2, one / norm, X2, incx2);
        dorbdb6(m1, m2, n, X1, incx1, X2, incx2, Q1, ldq1, Q2, ldq2,
                work, lwork, &childinfo);

        if (cblas_dnrm2(m1, X1, incx1) != zero ||
            cblas_dnrm2(m2, X2, incx2) != zero) {
            return;
        }
    }

    for (i = 0; i < m1; i++) {
        for (j = 0; j < m1; j++) {
            X1[j * incx1] = zero;
        }
        X1[i * incx1] = one;
        for (j = 0; j < m2; j++) {
            X2[j * incx2] = zero;
        }
        dorbdb6(m1, m2, n, X1, incx1, X2, incx2, Q1, ldq1, Q2, ldq2,
                work, lwork, &childinfo);
        if (cblas_dnrm2(m1, X1, incx1) != zero ||
            cblas_dnrm2(m2, X2, incx2) != zero) {
            return;
        }
    }

    for (i = 0; i < m2; i++) {
        for (j = 0; j < m1; j++) {
            X1[j * incx1] = zero;
        }
        for (j = 0; j < m2; j++) {
            X2[j * incx2] = zero;
        }
        X2[i * incx2] = one;
        dorbdb6(m1, m2, n, X1, incx1, X2, incx2, Q1, ldq1, Q2, ldq2,
                work, lwork, &childinfo);
        if (cblas_dnrm2(m1, X1, incx1) != zero ||
            cblas_dnrm2(m2, X2, incx2) != zero) {
            return;
        }
    }
}
