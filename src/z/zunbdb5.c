/**
 * @file zunbdb5.c
 * @brief ZUNBDB5 orthogonalizes the column vector X with respect to the
 *        columns of Q.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/**
 * ZUNBDB5 orthogonalizes the column vector
 *      X = [ X1 ]
 *          [ X2 ]
 * with respect to the columns of
 *      Q = [ Q1 ] .
 *          [ Q2 ]
 * The columns of Q must be orthonormal.
 *
 * If the projection is zero according to Kahan's "twice is enough"
 * criterion, then some other vector from the orthogonal complement
 * is returned. This vector is chosen in an arbitrary but deterministic
 * way.
 *
 * @param[in]     m1      The dimension of X1 and the number of rows in Q1. 0 <= M1.
 * @param[in]     m2      The dimension of X2 and the number of rows in Q2. 0 <= M2.
 * @param[in]     n       The number of columns in Q1 and Q2. 0 <= N.
 * @param[in,out] X1      Complex*16 array, dimension (M1).
 *                        On entry, the top part of the vector to be orthogonalized.
 *                        On exit, the top part of the projected vector.
 * @param[in]     incx1   Increment for entries of X1.
 * @param[in,out] X2      Complex*16 array, dimension (M2).
 *                        On entry, the bottom part of the vector to be
 *                        orthogonalized. On exit, the bottom part of the projected
 *                        vector.
 * @param[in]     incx2   Increment for entries of X2.
 * @param[in]     Q1      Complex*16 array, dimension (LDQ1, N).
 *                        The top part of the orthonormal basis matrix.
 * @param[in]     ldq1    The leading dimension of Q1. LDQ1 >= M1.
 * @param[in]     Q2      Complex*16 array, dimension (LDQ2, N).
 *                        The bottom part of the orthonormal basis matrix.
 * @param[in]     ldq2    The leading dimension of Q2. LDQ2 >= M2.
 * @param[out]    work    Complex*16 array, dimension (LWORK).
 * @param[in]     lwork   The dimension of the array WORK. LWORK >= N.
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if INFO = -i, the i-th argument had an illegal value.
 */
void zunbdb5(const INT m1, const INT m2, const INT n,
             c128* restrict X1, const INT incx1,
             c128* restrict X2, const INT incx2,
             c128* restrict Q1, const INT ldq1,
             c128* restrict Q2, const INT ldq2,
             c128* restrict work, const INT lwork,
             INT* info)
{
    const f64 REALZERO = 0.0;
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

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
        xerbla("ZUNBDB5", -(*info));
        return;
    }

    eps = dlamch("P");

    /* Project X onto the orthogonal complement of Q if X is nonzero */
    scl = REALZERO;
    ssq = REALZERO;
    zlassq(m1, X1, incx1, &scl, &ssq);
    zlassq(m2, X2, incx2, &scl, &ssq);
    norm = scl * sqrt(ssq);

    if (norm > n * eps) {
        /*  Scale vector to unit norm to avoid problems in the caller code.
         *  Computing the reciprocal is undesirable but
         *   * xLASCL cannot be used because of the vector increments and
         *   * the round-off error has a negligible impact on
         *     orthogonalization. */
        c128 inv_norm = ONE / norm;
        cblas_zscal(m1, &inv_norm, X1, incx1);
        cblas_zscal(m2, &inv_norm, X2, incx2);
        zunbdb6(m1, m2, n, X1, incx1, X2, incx2, Q1, ldq1, Q2,
                ldq2, work, lwork, &childinfo);

        /* If the projection is nonzero, then return */
        if (cblas_dznrm2(m1, X1, incx1) != REALZERO
            || cblas_dznrm2(m2, X2, incx2) != REALZERO) {
            return;
        }
    }

    /* Project each standard basis vector e_1,...,e_M1 in turn, stopping
     * when a nonzero projection is found */
    for (i = 0; i < m1; i++) {
        for (j = 0; j < m1; j++) {
            X1[j] = ZERO;
        }
        X1[i] = ONE;
        for (j = 0; j < m2; j++) {
            X2[j] = ZERO;
        }
        zunbdb6(m1, m2, n, X1, incx1, X2, incx2, Q1, ldq1, Q2,
                ldq2, work, lwork, &childinfo);
        if (cblas_dznrm2(m1, X1, incx1) != REALZERO
            || cblas_dznrm2(m2, X2, incx2) != REALZERO) {
            return;
        }
    }

    /* Project each standard basis vector e_(M1+1),...,e_(M1+M2) in turn,
     * stopping when a nonzero projection is found */
    for (i = 0; i < m2; i++) {
        for (j = 0; j < m1; j++) {
            X1[j] = ZERO;
        }
        for (j = 0; j < m2; j++) {
            X2[j] = ZERO;
        }
        X2[i] = ONE;
        zunbdb6(m1, m2, n, X1, incx1, X2, incx2, Q1, ldq1, Q2,
                ldq2, work, lwork, &childinfo);
        if (cblas_dznrm2(m1, X1, incx1) != REALZERO
            || cblas_dznrm2(m2, X2, incx2) != REALZERO) {
            return;
        }
    }
}
