/**
 * @file cunbdb6.c
 * @brief CUNBDB6 orthogonalizes a column vector with respect to an
 *        orthonormal basis using Gram-Schmidt.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"
#include <math.h>

/**
 * CUNBDB6 orthogonalizes the column vector
 *      X = [ X1 ]
 *          [ X2 ]
 * with respect to the columns of
 *      Q = [ Q1 ] .
 *          [ Q2 ]
 * The columns of Q must be orthonormal. The orthogonalized vector will
 * be zero if and only if it lies entirely in the range of Q.
 *
 * The projection is computed with at most two iterations of the
 * classical Gram-Schmidt algorithm, see
 * * L. Giraud, J. Langou, M. Rozloznik. "On the round-off error
 *   analysis of the Gram-Schmidt algorithm with reorthogonalization."
 *   2002. CERFACS Technical Report No. TR/PA/02/33.
 *
 * @param[in]     m1     The dimension of X1 and the number of rows in Q1.
 *                       0 <= M1.
 * @param[in]     m2     The dimension of X2 and the number of rows in Q2.
 *                       0 <= M2.
 * @param[in]     n      The number of columns in Q1 and Q2. 0 <= N.
 * @param[in,out] X1     Complex array, dimension (M1).
 *                       On entry, the top part of the vector to be orthogonalized.
 *                       On exit, the top part of the projected vector.
 * @param[in]     incx1  Increment for entries of X1.
 * @param[in,out] X2     Complex array, dimension (M2).
 *                       On entry, the bottom part of the vector to be
 *                       orthogonalized. On exit, the bottom part of the projected
 *                       vector.
 * @param[in]     incx2  Increment for entries of X2.
 * @param[in]     Q1     Complex array, dimension (LDQ1, N).
 *                       The top part of the orthonormal basis matrix.
 * @param[in]     ldq1   The leading dimension of Q1. LDQ1 >= M1.
 * @param[in]     Q2     Complex array, dimension (LDQ2, N).
 *                       The bottom part of the orthonormal basis matrix.
 * @param[in]     ldq2   The leading dimension of Q2. LDQ2 >= M2.
 * @param[out]    work   Complex array, dimension (LWORK).
 * @param[in]     lwork  The dimension of the array WORK. LWORK >= N.
 * @param[out]    info   = 0: successful exit.
 *                       < 0: if info = -i, the i-th argument had an illegal value.
 */
void cunbdb6(const INT m1, const INT m2, const INT n,
             c64* restrict X1, const INT incx1,
             c64* restrict X2, const INT incx2,
             const c64* restrict Q1, const INT ldq1,
             const c64* restrict Q2, const INT ldq2,
             c64* restrict work, const INT lwork,
             INT* info)
{
    const f32 ALPHA = 0.83f;
    const f32 REALZERO = 0.0f;
    const c64 NEGONE = CMPLXF(-1.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

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
        xerbla("CUNBDB6", -(*info));
        return;
    }

    eps = slamch("P");

    /* Compute the Euclidean norm of X */
    scl = REALZERO;
    ssq = REALZERO;
    classq(m1, X1, incx1, &scl, &ssq);
    classq(m2, X2, incx2, &scl, &ssq);
    norm = scl * sqrtf(ssq);

    /* First, project X onto the orthogonal complement of Q's column
       space */
    if (m1 == 0) {
        for (i = 0; i < n; i++) {
            work[i] = ZERO;
        }
    } else {
        cblas_cgemv(CblasColMajor, CblasConjTrans, m1, n, &ONE, Q1, ldq1,
                    X1, incx1, &ZERO, work, 1);
    }

    cblas_cgemv(CblasColMajor, CblasConjTrans, m2, n, &ONE, Q2, ldq2,
                X2, incx2, &ONE, work, 1);

    cblas_cgemv(CblasColMajor, CblasNoTrans, m1, n, &NEGONE, Q1, ldq1,
                work, 1, &ONE, X1, incx1);
    cblas_cgemv(CblasColMajor, CblasNoTrans, m2, n, &NEGONE, Q2, ldq2,
                work, 1, &ONE, X2, incx2);

    scl = REALZERO;
    ssq = REALZERO;
    classq(m1, X1, incx1, &scl, &ssq);
    classq(m2, X2, incx2, &scl, &ssq);
    norm_new = scl * sqrtf(ssq);

    /* If projection is sufficiently large in norm, then stop.
       If projection is zero, then stop.
       Otherwise, project again. */
    if (norm_new >= ALPHA * norm) {
        return;
    }

    if (norm_new <= n * eps * norm) {
        for (ix = 0; ix < 1 + (m1 - 1) * incx1; ix += incx1) {
            X1[ix] = ZERO;
        }
        for (ix = 0; ix < 1 + (m2 - 1) * incx2; ix += incx2) {
            X2[ix] = ZERO;
        }
        return;
    }

    norm = norm_new;

    for (i = 0; i < n; i++) {
        work[i] = ZERO;
    }

    if (m1 == 0) {
        for (i = 0; i < n; i++) {
            work[i] = ZERO;
        }
    } else {
        cblas_cgemv(CblasColMajor, CblasConjTrans, m1, n, &ONE, Q1, ldq1,
                    X1, incx1, &ZERO, work, 1);
    }

    cblas_cgemv(CblasColMajor, CblasConjTrans, m2, n, &ONE, Q2, ldq2,
                X2, incx2, &ONE, work, 1);

    cblas_cgemv(CblasColMajor, CblasNoTrans, m1, n, &NEGONE, Q1, ldq1,
                work, 1, &ONE, X1, incx1);
    cblas_cgemv(CblasColMajor, CblasNoTrans, m2, n, &NEGONE, Q2, ldq2,
                work, 1, &ONE, X2, incx2);

    scl = REALZERO;
    ssq = REALZERO;
    classq(m1, X1, incx1, &scl, &ssq);
    classq(m2, X2, incx2, &scl, &ssq);
    norm_new = scl * sqrtf(ssq);

    /* If second projection is sufficiently large in norm, then do
       nothing more. Alternatively, if it shrunk significantly, then
       truncate it to zero. */
    if (norm_new < ALPHA * norm) {
        for (ix = 0; ix < 1 + (m1 - 1) * incx1; ix += incx1) {
            X1[ix] = ZERO;
        }
        for (ix = 0; ix < 1 + (m2 - 1) * incx2; ix += incx2) {
            X2[ix] = ZERO;
        }
    }

    return;
}
