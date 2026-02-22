/**
 * @file ctrsyl.c
 * @brief CTRSYL solves the complex Sylvester matrix equation.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

/**
 * CTRSYL solves the complex Sylvester matrix equation:
 *
 *    op(A)*X + X*op(B) = scale*C or
 *    op(A)*X - X*op(B) = scale*C,
 *
 * where op(A) = A or A**H, and A and B are both upper triangular. A is
 * M-by-M and B is N-by-N; the right hand side C and the solution X are
 * M-by-N; and scale is an output scale factor, set <= 1 to avoid
 * overflow in X.
 *
 * @param[in] trana  Specifies the option op(A):
 *                   = 'N': op(A) = A    (No transpose)
 *                   = 'C': op(A) = A**H (Conjugate transpose)
 * @param[in] tranb  Specifies the option op(B):
 *                   = 'N': op(B) = B    (No transpose)
 *                   = 'C': op(B) = B**H (Conjugate transpose)
 * @param[in] isgn   Specifies the sign in the equation:
 *                   = +1: solve op(A)*X + X*op(B) = scale*C
 *                   = -1: solve op(A)*X - X*op(B) = scale*C
 * @param[in] m      The order of the matrix A, and the number of rows in the
 *                   matrices X and C. m >= 0.
 * @param[in] n      The order of the matrix B, and the number of columns in the
 *                   matrices X and C. n >= 0.
 * @param[in] A      Single complex array, dimension (lda, m).
 *                   The upper triangular matrix A.
 * @param[in] lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[in] B      Single complex array, dimension (ldb, n).
 *                   The upper triangular matrix B.
 * @param[in] ldb    The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] C  Single complex array, dimension (ldc, n).
 *                   On entry, the M-by-N right hand side matrix C.
 *                   On exit, C is overwritten by the solution matrix X.
 * @param[in] ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out] scale The scale factor, scale, set <= 1 to avoid overflow in X.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: A and B have common or very close eigenvalues; perturbed
 *                           values were used to solve the equation (but the matrices
 *                           A and B are unchanged).
 */
void ctrsyl(const char* trana, const char* tranb, const INT isgn,
            const INT m, const INT n,
            const c64* A, const INT lda,
            const c64* B, const INT ldb,
            c64* C, const INT ldc,
            f32* scale, INT* info)
{
    const f32 ONE = 1.0f;

    INT notrna, notrnb;
    INT j, k, l;
    f32 bignum, da11, db, eps, scaloc, sgn, smin, smlnum;
    c64 a11, suml, sumr, vec, x11;
    f32 dum[1];

    /* Decode and Test input parameters */
    notrna = (trana[0] == 'N' || trana[0] == 'n');
    notrnb = (tranb[0] == 'N' || tranb[0] == 'n');

    *info = 0;
    if (!notrna && !(trana[0] == 'C' || trana[0] == 'c')) {
        *info = -1;
    } else if (!notrnb && !(tranb[0] == 'C' || tranb[0] == 'c')) {
        *info = -2;
    } else if (isgn != 1 && isgn != -1) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("CTRSYL", -(*info));
        return;
    }

    /* Quick return if possible */
    *scale = ONE;
    if (m == 0 || n == 0)
        return;

    /* Set constants to control overflow */
    eps = slamch("P");
    smlnum = slamch("S");
    smlnum = smlnum * (f32)(m * n) / eps;
    bignum = ONE / smlnum;

    smin = smlnum;
    dum[0] = clange("M", m, m, A, lda, dum);
    if (eps * dum[0] > smin) smin = eps * dum[0];
    dum[0] = clange("M", n, n, B, ldb, dum);
    if (eps * dum[0] > smin) smin = eps * dum[0];

    sgn = (f32)isgn;

    if (notrna && notrnb) {
        /*
         * Solve    A*X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-left corner column by column by
         *
         *     A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *                 M                        L-1
         *       R(K,L) = SUM [A(K,I)*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)].
         *               I=K+1                      J=1
         */
        for (l = 0; l < n; l++) {
            for (k = m - 1; k >= 0; k--) {

                cblas_cdotu_sub(m - k - 1,
                                &A[k + ((k < m - 1 ? k + 1 : m - 1)) * lda], lda,
                                &C[(k < m - 1 ? k + 1 : m - 1) + l * ldc], 1,
                                &suml);
                cblas_cdotu_sub(l,
                                &C[k], ldc,
                                &B[l * ldb], 1,
                                &sumr);
                vec = C[k + l * ldc] - (suml + sgn * sumr);

                scaloc = ONE;
                a11 = A[k + k * lda] + sgn * B[l + l * ldb];
                da11 = fabsf(crealf(a11)) + fabsf(cimagf(a11));
                if (da11 <= smin) {
                    a11 = CMPLXF(smin, 0.0f);
                    da11 = smin;
                    *info = 1;
                }
                db = fabsf(crealf(vec)) + fabsf(cimagf(vec));
                if (da11 < ONE && db > ONE) {
                    if (db > bignum * da11)
                        scaloc = ONE / db;
                }
                x11 = cladiv(vec * CMPLXF(scaloc, 0.0f), a11);

                if (scaloc != ONE) {
                    for (j = 0; j < n; j++)
                        cblas_csscal(m, scaloc, &C[j * ldc], 1);
                    *scale = *scale * scaloc;
                }
                C[k + l * ldc] = x11;
            }
        }

    } else if (!notrna && notrnb) {
        /*
         * Solve    A**H *X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * upper-left corner column by column by
         *
         *     A**H(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *                K-1                           L-1
         *       R(K,L) = SUM [A**H(I,K)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)]
         *                I=1                           J=1
         */
        for (l = 0; l < n; l++) {
            for (k = 0; k < m; k++) {

                cblas_cdotc_sub(k,
                                &A[k * lda], 1,
                                &C[l * ldc], 1,
                                &suml);
                cblas_cdotu_sub(l,
                                &C[k], ldc,
                                &B[l * ldb], 1,
                                &sumr);
                vec = C[k + l * ldc] - (suml + sgn * sumr);

                scaloc = ONE;
                a11 = conjf(A[k + k * lda]) + sgn * B[l + l * ldb];
                da11 = fabsf(crealf(a11)) + fabsf(cimagf(a11));
                if (da11 <= smin) {
                    a11 = CMPLXF(smin, 0.0f);
                    da11 = smin;
                    *info = 1;
                }
                db = fabsf(crealf(vec)) + fabsf(cimagf(vec));
                if (da11 < ONE && db > ONE) {
                    if (db > bignum * da11)
                        scaloc = ONE / db;
                }

                x11 = cladiv(vec * CMPLXF(scaloc, 0.0f), a11);

                if (scaloc != ONE) {
                    for (j = 0; j < n; j++)
                        cblas_csscal(m, scaloc, &C[j * ldc], 1);
                    *scale = *scale * scaloc;
                }
                C[k + l * ldc] = x11;
            }
        }

    } else if (!notrna && !notrnb) {
        /*
         * Solve    A**H*X + ISGN*X*B**H = C.
         *
         * The (K,L)th block of X is determined starting from
         * upper-right corner column by column by
         *
         *     A**H(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *                 K-1
         *        R(K,L) = SUM [A**H(I,K)*X(I,L)] +
         *                 I=1
         *                        N
         *                  ISGN*SUM [X(K,J)*B**H(L,J)].
         *                       J=L+1
         */
        for (l = n - 1; l >= 0; l--) {
            for (k = 0; k < m; k++) {

                cblas_cdotc_sub(k,
                                &A[k * lda], 1,
                                &C[l * ldc], 1,
                                &suml);
                cblas_cdotc_sub(n - l - 1,
                                &C[k + (l < n - 1 ? l + 1 : n - 1) * ldc], ldc,
                                &B[l + (l < n - 1 ? l + 1 : n - 1) * ldb], ldb,
                                &sumr);
                vec = C[k + l * ldc] - (suml + sgn * conjf(sumr));

                scaloc = ONE;
                a11 = conjf(A[k + k * lda] + sgn * B[l + l * ldb]);
                da11 = fabsf(crealf(a11)) + fabsf(cimagf(a11));
                if (da11 <= smin) {
                    a11 = CMPLXF(smin, 0.0f);
                    da11 = smin;
                    *info = 1;
                }
                db = fabsf(crealf(vec)) + fabsf(cimagf(vec));
                if (da11 < ONE && db > ONE) {
                    if (db > bignum * da11)
                        scaloc = ONE / db;
                }

                x11 = cladiv(vec * CMPLXF(scaloc, 0.0f), a11);

                if (scaloc != ONE) {
                    for (j = 0; j < n; j++)
                        cblas_csscal(m, scaloc, &C[j * ldc], 1);
                    *scale = *scale * scaloc;
                }
                C[k + l * ldc] = x11;
            }
        }

    } else if (notrna && !notrnb) {
        /*
         * Solve    A*X + ISGN*X*B**H = C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-left corner column by column by
         *
         *    A(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *                 M                          N
         *       R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B**H(L,J)]
         *               I=K+1                      J=L+1
         */
        for (l = n - 1; l >= 0; l--) {
            for (k = m - 1; k >= 0; k--) {

                cblas_cdotu_sub(m - k - 1,
                                &A[k + (k < m - 1 ? k + 1 : m - 1) * lda], lda,
                                &C[(k < m - 1 ? k + 1 : m - 1) + l * ldc], 1,
                                &suml);
                cblas_cdotc_sub(n - l - 1,
                                &C[k + (l < n - 1 ? l + 1 : n - 1) * ldc], ldc,
                                &B[l + (l < n - 1 ? l + 1 : n - 1) * ldb], ldb,
                                &sumr);
                vec = C[k + l * ldc] - (suml + sgn * conjf(sumr));

                scaloc = ONE;
                a11 = A[k + k * lda] + sgn * conjf(B[l + l * ldb]);
                da11 = fabsf(crealf(a11)) + fabsf(cimagf(a11));
                if (da11 <= smin) {
                    a11 = CMPLXF(smin, 0.0f);
                    da11 = smin;
                    *info = 1;
                }
                db = fabsf(crealf(vec)) + fabsf(cimagf(vec));
                if (da11 < ONE && db > ONE) {
                    if (db > bignum * da11)
                        scaloc = ONE / db;
                }

                x11 = cladiv(vec * CMPLXF(scaloc, 0.0f), a11);

                if (scaloc != ONE) {
                    for (j = 0; j < n; j++)
                        cblas_csscal(m, scaloc, &C[j * ldc], 1);
                    *scale = *scale * scaloc;
                }
                C[k + l * ldc] = x11;
            }
        }
    }
}
