/**
 * @file dtrsyl.c
 * @brief DTRSYL solves the real Sylvester matrix equation.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

/**
 * DTRSYL solves the real Sylvester matrix equation:
 *
 *    op(A)*X + ISGN*X*op(B) = scale*C or
 *    op(A)*X - ISGN*X*op(B) = scale*C,
 *
 * where op(A) = A or A**T, and A and B are both upper quasi-
 * triangular. A is M-by-M and B is N-by-N; the right hand side C and
 * the solution X are M-by-N; and scale is an output scale factor, set
 * <= 1 to avoid overflow in X.
 *
 * A and B must be in Schur canonical form (as returned by DHSEQR), that
 * is, block upper triangular with 1-by-1 and 2-by-2 diagonal blocks;
 * each 2-by-2 diagonal block has its diagonal elements equal and its
 * off-diagonal elements of opposite sign.
 *
 * @param[in] trana  Specifies the option op(A):
 *                   = 'N': op(A) = A    (No transpose)
 *                   = 'T': op(A) = A**T (Transpose)
 *                   = 'C': op(A) = A**H (Conjugate transpose = Transpose)
 * @param[in] tranb  Specifies the option op(B):
 *                   = 'N': op(B) = B    (No transpose)
 *                   = 'T': op(B) = B**T (Transpose)
 *                   = 'C': op(B) = B**H (Conjugate transpose = Transpose)
 * @param[in] isgn   Specifies the sign in the equation:
 *                   = +1: solve op(A)*X + X*op(B) = scale*C
 *                   = -1: solve op(A)*X - X*op(B) = scale*C
 * @param[in] m      The order of the matrix A, and the number of rows in the
 *                   matrices X and C. m >= 0.
 * @param[in] n      The order of the matrix B, and the number of columns in the
 *                   matrices X and C. n >= 0.
 * @param[in] A      Double precision array, dimension (lda, m).
 *                   The upper quasi-triangular matrix A, in Schur canonical form.
 * @param[in] lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[in] B      Double precision array, dimension (ldb, n).
 *                   The upper quasi-triangular matrix B, in Schur canonical form.
 * @param[in] ldb    The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] C  Double precision array, dimension (ldc, n).
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
void dtrsyl(const char* trana, const char* tranb, const int isgn,
            const int m, const int n,
            const double* A, const int lda,
            const double* B, const int ldb,
            double* C, const int ldc,
            double* scale, int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int notrna, notrnb;
    int ierr, j, k, k1, k2, knext, l, l1, l2, lnext;
    double a11, bignum, da11, db, eps, scaloc, sgn, smin;
    double smlnum, suml, sumr, xnorm;
    double dum[1], vec[4], x[4];  /* vec and x are 2x2, column-major */
    int minval;

    /* Decode and test input parameters */
    notrna = (trana[0] == 'N' || trana[0] == 'n');
    notrnb = (tranb[0] == 'N' || tranb[0] == 'n');

    *info = 0;
    if (!notrna && !(trana[0] == 'T' || trana[0] == 't') &&
        !(trana[0] == 'C' || trana[0] == 'c')) {
        *info = -1;
    } else if (!notrnb && !(tranb[0] == 'T' || tranb[0] == 't') &&
               !(tranb[0] == 'C' || tranb[0] == 'c')) {
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
        return;
    }

    /* Quick return if possible */
    *scale = ONE;
    if (m == 0 || n == 0)
        return;

    /* Set constants to control overflow */
    eps = dlamch("P");
    smlnum = dlamch("S");
    bignum = ONE / smlnum;
    smlnum = smlnum * (double)(m * n) / eps;
    bignum = ONE / smlnum;

    smin = smlnum;
    dum[0] = dlange("M", m, m, A, lda, dum);
    if (eps * dum[0] > smin) smin = eps * dum[0];
    dum[0] = dlange("M", n, n, B, ldb, dum);
    if (eps * dum[0] > smin) smin = eps * dum[0];

    sgn = (double)isgn;

    if (notrna && notrnb) {
        /*
         * Solve    A*X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-left corner column by column by
         *
         *  A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *              M                         L-1
         *    R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)].
         *            I=K+1                       J=1
         *
         * Start column loop (index = L)
         * L1 (L2): column index of the first (last) row of X(K,L).
         */
        lnext = 0;
        for (l = 0; l < n; l++) {
            if (l < lnext)
                continue;
            if (l == n - 1) {
                l1 = l;
                l2 = l;
            } else {
                if (B[(l + 1) + l * ldb] != ZERO) {
                    l1 = l;
                    l2 = l + 1;
                    lnext = l + 2;
                } else {
                    l1 = l;
                    l2 = l;
                    lnext = l + 1;
                }
            }

            /*
             * Start row loop (index = K)
             * K1 (K2): row index of the first (last) row of X(K,L).
             */
            knext = m - 1;
            for (k = m - 1; k >= 0; k--) {
                if (k > knext)
                    continue;
                if (k == 0) {
                    k1 = k;
                    k2 = k;
                } else {
                    if (A[k + (k - 1) * lda] != ZERO) {
                        k1 = k - 1;
                        k2 = k;
                        knext = k - 2;
                    } else {
                        k1 = k;
                        k2 = k;
                        knext = k - 1;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    minval = k1 + 1 < m ? k1 + 1 : m - 1;
                    suml = cblas_ddot(m - k1 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);
                    scaloc = ONE;

                    a11 = A[k1 + k1 * lda] + sgn * B[l1 + l1 * ldb];
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(vec[0]);
                    if (da11 < ONE && db > ONE) {
                        if (db > bignum * da11)
                            scaloc = ONE / db;
                    }
                    x[0] = (vec[0] * scaloc) / a11;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];

                } else if (l1 == l2 && k1 != k2) {

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(m - k2 - 1, &A[k2 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k2], ldc, &B[l1 * ldb], 1);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    dlaln2(0, 2, 1, smin, ONE, &A[k1 + k1 * lda],
                           lda, ONE, ONE, vec, 2, -sgn * B[l1 + l1 * ldb],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k2 + l1 * ldc] = x[1];

                } else if (l1 != l2 && k1 == k2) {

                    minval = k1 + 1 < m ? k1 + 1 : m - 1;
                    suml = cblas_ddot(m - k1 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = sgn * (C[k1 + l1 * ldc] - (suml + sgn * sumr));

                    suml = cblas_ddot(m - k1 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l2 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l2 * ldb], 1);
                    vec[1] = sgn * (C[k1 + l2 * ldc] - (suml + sgn * sumr));

                    dlaln2(1, 2, 1, smin, ONE, &B[l1 + l1 * ldb],
                           ldb, ONE, ONE, vec, 2, -sgn * A[k1 + k1 * lda],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[1];

                } else if (l1 != l2 && k1 != k2) {

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(m - k2 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l2 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l2 * ldb], 1);
                    vec[2] = C[k1 + l2 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(m - k2 - 1, &A[k2 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k2], ldc, &B[l1 * ldb], 1);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(m - k2 - 1, &A[k2 + minval * lda], lda,
                                      &C[minval + l2 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k2], ldc, &B[l2 * ldb], 1);
                    vec[3] = C[k2 + l2 * ldc] - (suml + sgn * sumr);

                    dlasy2(0, 0, isgn, 2, 2,
                           &A[k1 + k1 * lda], lda, &B[l1 + l1 * ldb], ldb, vec,
                           2, &scaloc, x, 2, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[2];
                    C[k2 + l1 * ldc] = x[1];
                    C[k2 + l2 * ldc] = x[3];
                }
            }
        }

    } else if (!notrna && notrnb) {
        /*
         * Solve    A**T *X + ISGN*X*B = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * upper-left corner column by column by
         *
         *   A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
         *
         * Where
         *            K-1        T                    L-1
         *   R(K,L) = SUM [A(I,K)**T*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]
         *            I=0                          J=0
         *
         * Start column loop (index = L)
         * L1 (L2): column index of the first (last) row of X(K,L)
         */
        lnext = 0;
        for (l = 0; l < n; l++) {
            if (l < lnext)
                continue;
            if (l == n - 1) {
                l1 = l;
                l2 = l;
            } else {
                if (B[(l + 1) + l * ldb] != ZERO) {
                    l1 = l;
                    l2 = l + 1;
                    lnext = l + 2;
                } else {
                    l1 = l;
                    l2 = l;
                    lnext = l + 1;
                }
            }

            /*
             * Start row loop (index = K)
             * K1 (K2): row index of the first (last) row of X(K,L)
             */
            knext = 0;
            for (k = 0; k < m; k++) {
                if (k < knext)
                    continue;
                if (k == m - 1) {
                    k1 = k;
                    k2 = k;
                } else {
                    if (A[(k + 1) + k * lda] != ZERO) {
                        k1 = k;
                        k2 = k + 1;
                        knext = k + 2;
                    } else {
                        k1 = k;
                        k2 = k;
                        knext = k + 1;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);
                    scaloc = ONE;

                    a11 = A[k1 + k1 * lda] + sgn * B[l1 + l1 * ldb];
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(vec[0]);
                    if (da11 < ONE && db > ONE) {
                        if (db > bignum * da11)
                            scaloc = ONE / db;
                    }
                    x[0] = (vec[0] * scaloc) / a11;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];

                } else if (l1 == l2 && k1 != k2) {

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k2 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k2], ldc, &B[l1 * ldb], 1);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    dlaln2(1, 2, 1, smin, ONE, &A[k1 + k1 * lda],
                           lda, ONE, ONE, vec, 2, -sgn * B[l1 + l1 * ldb],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k2 + l1 * ldc] = x[1];

                } else if (l1 != l2 && k1 == k2) {

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = sgn * (C[k1 + l1 * ldc] - (suml + sgn * sumr));

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l2 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l2 * ldb], 1);
                    vec[1] = sgn * (C[k1 + l2 * ldc] - (suml + sgn * sumr));

                    dlaln2(1, 2, 1, smin, ONE, &B[l1 + l1 * ldb],
                           ldb, ONE, ONE, vec, 2, -sgn * A[k1 + k1 * lda],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[1];

                } else if (l1 != l2 && k1 != k2) {

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l1 * ldb], 1);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l2 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k1], ldc, &B[l2 * ldb], 1);
                    vec[2] = C[k1 + l2 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k2 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k2], ldc, &B[l1 * ldb], 1);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k2 * lda], 1, &C[l2 * ldc], 1);
                    sumr = cblas_ddot(l1, &C[k2], ldc, &B[l2 * ldb], 1);
                    vec[3] = C[k2 + l2 * ldc] - (suml + sgn * sumr);

                    dlasy2(1, 0, isgn, 2, 2,
                           &A[k1 + k1 * lda], lda, &B[l1 + l1 * ldb], ldb, vec,
                           2, &scaloc, x, 2, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[2];
                    C[k2 + l1 * ldc] = x[1];
                    C[k2 + l2 * ldc] = x[3];
                }
            }
        }

    } else if (!notrna && !notrnb) {
        /*
         * Solve    A**T*X + ISGN*X*B**T = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * top-right corner column by column by
         *
         *    A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
         *
         * Where
         *              K-1                            N-1
         *     R(K,L) = SUM [A(I,K)**T*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
         *              I=0                          J=L+1
         *
         * Start column loop (index = L)
         * L1 (L2): column index of the first (last) row of X(K,L)
         */
        lnext = n - 1;
        for (l = n - 1; l >= 0; l--) {
            if (l > lnext)
                continue;
            if (l == 0) {
                l1 = l;
                l2 = l;
            } else {
                if (B[l + (l - 1) * ldb] != ZERO) {
                    l1 = l - 1;
                    l2 = l;
                    lnext = l - 2;
                } else {
                    l1 = l;
                    l2 = l;
                    lnext = l - 1;
                }
            }

            /*
             * Start row loop (index = K)
             * K1 (K2): row index of the first (last) row of X(K,L)
             */
            knext = 0;
            for (k = 0; k < m; k++) {
                if (k < knext)
                    continue;
                if (k == m - 1) {
                    k1 = k;
                    k2 = k;
                } else {
                    if (A[(k + 1) + k * lda] != ZERO) {
                        k1 = k;
                        k2 = k + 1;
                        knext = k + 2;
                    } else {
                        k1 = k;
                        k2 = k;
                        knext = k + 1;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    minval = l1 + 1 < n ? l1 + 1 : n - 1;
                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(n - l1 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);
                    scaloc = ONE;

                    a11 = A[k1 + k1 * lda] + sgn * B[l1 + l1 * ldb];
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(vec[0]);
                    if (da11 < ONE && db > ONE) {
                        if (db > bignum * da11)
                            scaloc = ONE / db;
                    }
                    x[0] = (vec[0] * scaloc) / a11;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];

                } else if (l1 == l2 && k1 != k2) {

                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k2 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k2 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    dlaln2(1, 2, 1, smin, ONE, &A[k1 + k1 * lda],
                           lda, ONE, ONE, vec, 2, -sgn * B[l1 + l1 * ldb],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k2 + l1 * ldc] = x[1];

                } else if (l1 != l2 && k1 == k2) {

                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = sgn * (C[k1 + l1 * ldc] - (suml + sgn * sumr));

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l2 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l2 + minval * ldb], ldb);
                    vec[1] = sgn * (C[k1 + l2 * ldc] - (suml + sgn * sumr));

                    dlaln2(0, 2, 1, smin, ONE, &B[l1 + l1 * ldb],
                           ldb, ONE, ONE, vec, 2, -sgn * A[k1 + k1 * lda],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[1];

                } else if (l1 != l2 && k1 != k2) {

                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k1 * lda], 1, &C[l2 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l2 + minval * ldb], ldb);
                    vec[2] = C[k1 + l2 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k2 * lda], 1, &C[l1 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k2 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    suml = cblas_ddot(k1, &A[k2 * lda], 1, &C[l2 * ldc], 1);
                    sumr = cblas_ddot(n - l2 - 1, &C[k2 + minval * ldc], ldc,
                                      &B[l2 + minval * ldb], ldb);
                    vec[3] = C[k2 + l2 * ldc] - (suml + sgn * sumr);

                    dlasy2(1, 1, isgn, 2, 2,
                           &A[k1 + k1 * lda], lda, &B[l1 + l1 * ldb], ldb, vec,
                           2, &scaloc, x, 2, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[2];
                    C[k2 + l1 * ldc] = x[1];
                    C[k2 + l2 * ldc] = x[3];
                }
            }
        }

    } else if (notrna && !notrnb) {
        /*
         * Solve    A*X + ISGN*X*B**T = scale*C.
         *
         * The (K,L)th block of X is determined starting from
         * bottom-right corner column by column by
         *
         *     A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
         *
         * Where
         *               M-1                          N-1
         *     R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
         *             I=K+1                      J=L+1
         *
         * Start column loop (index = L)
         * L1 (L2): column index of the first (last) row of X(K,L)
         */
        lnext = n - 1;
        for (l = n - 1; l >= 0; l--) {
            if (l > lnext)
                continue;
            if (l == 0) {
                l1 = l;
                l2 = l;
            } else {
                if (B[l + (l - 1) * ldb] != ZERO) {
                    l1 = l - 1;
                    l2 = l;
                    lnext = l - 2;
                } else {
                    l1 = l;
                    l2 = l;
                    lnext = l - 1;
                }
            }

            /*
             * Start row loop (index = K)
             * K1 (K2): row index of the first (last) row of X(K,L)
             */
            knext = m - 1;
            for (k = m - 1; k >= 0; k--) {
                if (k > knext)
                    continue;
                if (k == 0) {
                    k1 = k;
                    k2 = k;
                } else {
                    if (A[k + (k - 1) * lda] != ZERO) {
                        k1 = k - 1;
                        k2 = k;
                        knext = k - 2;
                    } else {
                        k1 = k;
                        k2 = k;
                        knext = k - 1;
                    }
                }

                if (l1 == l2 && k1 == k2) {
                    minval = k1 + 1 < m ? k1 + 1 : m - 1;
                    suml = cblas_ddot(m - k1 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    minval = l1 + 1 < n ? l1 + 1 : n - 1;
                    sumr = cblas_ddot(n - l1 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);
                    scaloc = ONE;

                    a11 = A[k1 + k1 * lda] + sgn * B[l1 + l1 * ldb];
                    da11 = fabs(a11);
                    if (da11 <= smin) {
                        a11 = smin;
                        da11 = smin;
                        *info = 1;
                    }
                    db = fabs(vec[0]);
                    if (da11 < ONE && db > ONE) {
                        if (db > bignum * da11)
                            scaloc = ONE / db;
                    }
                    x[0] = (vec[0] * scaloc) / a11;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];

                } else if (l1 == l2 && k1 != k2) {

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k2 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k2 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    dlaln2(0, 2, 1, smin, ONE, &A[k1 + k1 * lda],
                           lda, ONE, ONE, vec, 2, -sgn * B[l1 + l1 * ldb],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k2 + l1 * ldc] = x[1];

                } else if (l1 != l2 && k1 == k2) {

                    minval = k1 + 1 < m ? k1 + 1 : m - 1;
                    suml = cblas_ddot(m - k1 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = sgn * (C[k1 + l1 * ldc] - (suml + sgn * sumr));

                    minval = k1 + 1 < m ? k1 + 1 : m - 1;
                    suml = cblas_ddot(m - k1 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l2 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l2 + minval * ldb], ldb);
                    vec[1] = sgn * (C[k1 + l2 * ldc] - (suml + sgn * sumr));

                    dlaln2(0, 2, 1, smin, ONE, &B[l1 + l1 * ldb],
                           ldb, ONE, ONE, vec, 2, -sgn * A[k1 + k1 * lda],
                           ZERO, x, 2, &scaloc, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[1];

                } else if (l1 != l2 && k1 != k2) {

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[0] = C[k1 + l1 * ldc] - (suml + sgn * sumr);

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k1 + minval * lda], lda,
                                      &C[minval + l2 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k1 + minval * ldc], ldc,
                                      &B[l2 + minval * ldb], ldb);
                    vec[2] = C[k1 + l2 * ldc] - (suml + sgn * sumr);

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k2 + minval * lda], lda,
                                      &C[minval + l1 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k2 + minval * ldc], ldc,
                                      &B[l1 + minval * ldb], ldb);
                    vec[1] = C[k2 + l1 * ldc] - (suml + sgn * sumr);

                    minval = k2 + 1 < m ? k2 + 1 : m - 1;
                    suml = cblas_ddot(m - k2 - 1, &A[k2 + minval * lda], lda,
                                      &C[minval + l2 * ldc], 1);
                    minval = l2 + 1 < n ? l2 + 1 : n - 1;
                    sumr = cblas_ddot(n - l2 - 1, &C[k2 + minval * ldc], ldc,
                                      &B[l2 + minval * ldb], ldb);
                    vec[3] = C[k2 + l2 * ldc] - (suml + sgn * sumr);

                    dlasy2(0, 1, isgn, 2, 2,
                           &A[k1 + k1 * lda], lda, &B[l1 + l1 * ldb], ldb, vec,
                           2, &scaloc, x, 2, &xnorm, &ierr);
                    if (ierr != 0)
                        *info = 1;

                    if (scaloc != ONE) {
                        for (j = 0; j < n; j++)
                            cblas_dscal(m, scaloc, &C[j * ldc], 1);
                        *scale = *scale * scaloc;
                    }
                    C[k1 + l1 * ldc] = x[0];
                    C[k1 + l2 * ldc] = x[2];
                    C[k2 + l1 * ldc] = x[1];
                    C[k2 + l2 * ldc] = x[3];
                }
            }
        }
    }
}
