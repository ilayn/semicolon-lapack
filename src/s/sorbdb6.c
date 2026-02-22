/**
 * @file sorbdb6.c
 * @brief SORBDB6 orthogonalizes a column vector with respect to the columns of a matrix using Gram-Schmidt.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SORBDB6 orthogonalizes the column vector
 *      X = [ X1 ]
 *          [ X2 ]
 * with respect to the columns of
 *      Q = [ Q1 ] .
 *          [ Q2 ]
 * The columns of Q must be orthonormal. The orthogonalized vector will
 * be zero if and only if it lies entirely in the range of Q.
 *
 * The projection is computed with at most two iterations of the
 * classical Gram-Schmidt algorithm.
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
void sorbdb6(
    const INT m1,
    const INT m2,
    const INT n,
    f32* restrict X1,
    const INT incx1,
    f32* restrict X2,
    const INT incx2,
    const f32* restrict Q1,
    const INT ldq1,
    const f32* restrict Q2,
    const INT ldq2,
    f32* restrict work,
    const INT lwork,
    INT* info)
{
    const f32 alpha = 0.83f;
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const f32 negone = -1.0f;
    INT i, ix;
    f32 eps, norm, norm_new, scl, ssq;

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
        xerbla("SORBDB6", -(*info));
        return;
    }

    eps = slamch("P");

    scl = zero;
    ssq = zero;
    slassq(m1, X1, incx1, &scl, &ssq);
    slassq(m2, X2, incx2, &scl, &ssq);
    norm = scl * sqrtf(ssq);

    if (m1 == 0) {
        for (i = 0; i < n; i++) {
            work[i] = zero;
        }
    } else {
        cblas_sgemv(CblasColMajor, CblasTrans, m1, n, one, Q1, ldq1, X1, incx1,
                    zero, work, 1);
    }

    cblas_sgemv(CblasColMajor, CblasTrans, m2, n, one, Q2, ldq2, X2, incx2,
                one, work, 1);

    cblas_sgemv(CblasColMajor, CblasNoTrans, m1, n, negone, Q1, ldq1, work, 1,
                one, X1, incx1);
    cblas_sgemv(CblasColMajor, CblasNoTrans, m2, n, negone, Q2, ldq2, work, 1,
                one, X2, incx2);

    scl = zero;
    ssq = zero;
    slassq(m1, X1, incx1, &scl, &ssq);
    slassq(m2, X2, incx2, &scl, &ssq);
    norm_new = scl * sqrtf(ssq);

    if (norm_new >= alpha * norm) {
        return;
    }

    if (norm_new <= n * eps * norm) {
        for (ix = 0; ix < m1; ix++) {
            X1[ix * incx1] = zero;
        }
        for (ix = 0; ix < m2; ix++) {
            X2[ix * incx2] = zero;
        }
        return;
    }

    norm = norm_new;

    for (i = 0; i < n; i++) {
        work[i] = zero;
    }

    if (m1 == 0) {
        for (i = 0; i < n; i++) {
            work[i] = zero;
        }
    } else {
        cblas_sgemv(CblasColMajor, CblasTrans, m1, n, one, Q1, ldq1, X1, incx1,
                    zero, work, 1);
    }

    cblas_sgemv(CblasColMajor, CblasTrans, m2, n, one, Q2, ldq2, X2, incx2,
                one, work, 1);

    cblas_sgemv(CblasColMajor, CblasNoTrans, m1, n, negone, Q1, ldq1, work, 1,
                one, X1, incx1);
    cblas_sgemv(CblasColMajor, CblasNoTrans, m2, n, negone, Q2, ldq2, work, 1,
                one, X2, incx2);

    scl = zero;
    ssq = zero;
    slassq(m1, X1, incx1, &scl, &ssq);
    slassq(m2, X2, incx2, &scl, &ssq);
    norm_new = scl * sqrtf(ssq);

    if (norm_new < alpha * norm) {
        for (ix = 0; ix < m1; ix++) {
            X1[ix * incx1] = zero;
        }
        for (ix = 0; ix < m2; ix++) {
            X2[ix * incx2] = zero;
        }
    }
}
