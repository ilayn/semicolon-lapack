/**
 * @file stfsm.c
 * @brief STFSM solves a matrix equation (one operand is a triangular matrix in RFP format).
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * Level 3 BLAS like routine for A in RFP Format.
 *
 * STFSM solves the matrix equation
 *
 *    op( A )*X = alpha*B  or  X*op( A ) = alpha*B
 *
 * where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 * non-unit, upper or lower triangular matrix and op( A ) is one of
 *
 *    op( A ) = A   or   op( A ) = A**T.
 *
 * A is in Rectangular Full Packed (RFP) Format.
 *
 * The matrix X is overwritten on B.
 *
 * @param[in] transr
 *          = 'N':  The Normal Form of RFP A is stored;
 *          = 'T':  The Transpose Form of RFP A is stored.
 *
 * @param[in] side
 *          On entry, SIDE specifies whether op( A ) appears on the left
 *          or right of X as follows:
 *             SIDE = 'L' or 'l'   op( A )*X = alpha*B.
 *             SIDE = 'R' or 'r'   X*op( A ) = alpha*B.
 *
 * @param[in] uplo
 *          On entry, UPLO specifies whether the RFP matrix A came from
 *          an upper or lower triangular matrix as follows:
 *          UPLO = 'U' or 'u' RFP A came from an upper triangular matrix
 *          UPLO = 'L' or 'l' RFP A came from a lower triangular matrix
 *
 * @param[in] trans
 *          On entry, TRANS specifies the form of op( A ) to be used
 *          in the matrix multiplication as follows:
 *             TRANS = 'N' or 'n'   op( A ) = A.
 *             TRANS = 'T' or 't'   op( A ) = A'.
 *
 * @param[in] diag
 *          On entry, DIAG specifies whether or not RFP A is unit
 *          triangular as follows:
 *             DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *             DIAG = 'N' or 'n'   A is not assumed to be unit triangular.
 *
 * @param[in] m
 *          On entry, M specifies the number of rows of B. M must be at
 *          least zero.
 *
 * @param[in] n
 *          On entry, N specifies the number of columns of B. N must be
 *          at least zero.
 *
 * @param[in] alpha
 *          On entry, ALPHA specifies the scalar alpha. When alpha is
 *          zero then A is not referenced and B need not be set before
 *          entry.
 *
 * @param[in] A
 *          Double precision array, dimension (NT)
 *          NT = N*(N+1)/2 if SIDE='R' and NT = M*(M+1)/2 otherwise.
 *          On entry, the matrix A in RFP Format.
 *
 * @param[in,out] B
 *          Double precision array, dimension (LDB,N)
 *          Before entry, the leading m by n part of the array B must
 *          contain the right-hand side matrix B, and on exit is
 *          overwritten by the solution matrix X.
 *
 * @param[in] ldb
 *          On entry, LDB specifies the first dimension of B as declared
 *          in the calling (sub) program. LDB must be at least max( 1, m ).
 */
void stfsm(
    const char* transr,
    const char* side,
    const char* uplo,
    const char* trans,
    const char* diag,
    const int m,
    const int n,
    const f32 alpha,
    const f32* const restrict A,
    f32* const restrict B,
    const int ldb)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    int lower, lside, misodd, nisodd, normaltransr, notrans;
    int m1, m2, n1, n2, k, info;

    CBLAS_DIAG cblas_siag;

    info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lside = (side[0] == 'L' || side[0] == 'l');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    notrans = (trans[0] == 'N' || trans[0] == 'n');

    if (!normaltransr && !(transr[0] == 'T' || transr[0] == 't')) {
        info = -1;
    } else if (!lside && !(side[0] == 'R' || side[0] == 'r')) {
        info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        info = -3;
    } else if (!notrans && !(trans[0] == 'T' || trans[0] == 't')) {
        info = -4;
    } else if (!(diag[0] == 'N' || diag[0] == 'n') &&
               !(diag[0] == 'U' || diag[0] == 'u')) {
        info = -5;
    } else if (m < 0) {
        info = -6;
    } else if (n < 0) {
        info = -7;
    } else if (ldb < (m > 1 ? m : 1)) {
        info = -11;
    }
    if (info != 0) {
        xerbla("STFSM ", -info);
        return;
    }

    if ((m == 0) || (n == 0)) {
        return;
    }

    if (alpha == ZERO) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                B[i + j * ldb] = ZERO;
            }
        }
        return;
    }

    cblas_siag = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;

    if (lside) {

        if (m % 2 == 0) {
            misodd = 0;
            k = m / 2;
        } else {
            misodd = 1;
            if (lower) {
                m2 = m / 2;
                m1 = m - m2;
            } else {
                m1 = m / 2;
                m2 = m - m1;
            }
        }

        if (misodd) {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        if (m == 1) {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                        m1, n, alpha, A, m, B, ldb);
                        } else {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                        m1, n, alpha, &A[0], m, B, ldb);
                            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                        m2, n, m1, -ONE, &A[m1], m, B, ldb, alpha, &B[m1], ldb);
                            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                        m2, n, ONE, &A[m], m, &B[m1], ldb);
                        }

                    } else {

                        if (m == 1) {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                        m1, n, alpha, &A[0], m, B, ldb);
                        } else {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                        m2, n, alpha, &A[m], m, &B[m1], ldb);
                            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                        m1, n, m2, -ONE, &A[m1], m, &B[m1], ldb, alpha, B, ldb);
                            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                        m1, n, ONE, &A[0], m, B, ldb);
                        }

                    }

                } else {

                    if (!notrans) {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                    m1, n, alpha, &A[m2], m, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    m2, n, m1, -ONE, &A[0], m, B, ldb, alpha, &B[m1], ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                    m2, n, ONE, &A[m1], m, &B[m1], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                    m2, n, alpha, &A[m1], m, &B[m1], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m1, n, m2, -ONE, &A[0], m, &B[m1], ldb, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                    m1, n, ONE, &A[m2], m, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        if (m == 1) {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                        m1, n, alpha, &A[0], m1, B, ldb);
                        } else {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                        m1, n, alpha, &A[0], m1, B, ldb);
                            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                        m2, n, m1, -ONE, &A[m1 * m1], m1, B, ldb, alpha, &B[m1], ldb);
                            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                        m2, n, ONE, &A[1], m1, &B[m1], ldb);
                        }

                    } else {

                        if (m == 1) {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                        m1, n, alpha, &A[0], m1, B, ldb);
                        } else {
                            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                        m2, n, alpha, &A[1], m1, &B[m1], ldb);
                            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                        m1, n, m2, -ONE, &A[m1 * m1], m1, &B[m1], ldb, alpha, B, ldb);
                            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                        m1, n, ONE, &A[0], m1, B, ldb);
                        }

                    }

                } else {

                    if (!notrans) {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                    m1, n, alpha, &A[m2 * m2], m2, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m2, n, m1, -ONE, &A[0], m2, B, ldb, alpha, &B[m1], ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                    m2, n, ONE, &A[m1 * m2], m2, &B[m1], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                    m2, n, alpha, &A[m1 * m2], m2, &B[m1], ldb);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    m1, n, m2, -ONE, &A[0], m2, &B[m1], ldb, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                    m1, n, ONE, &A[m2 * m2], m2, B, ldb);

                    }

                }

            }

        } else {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                    k, n, alpha, &A[1], m + 1, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[k + 1], m + 1, B, ldb, alpha, &B[k], ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                    k, n, ONE, &A[0], m + 1, &B[k], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                    k, n, alpha, &A[0], m + 1, &B[k], ldb);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[k + 1], m + 1, &B[k], ldb, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                    k, n, ONE, &A[1], m + 1, B, ldb);

                    }

                } else {

                    if (!notrans) {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                    k, n, alpha, &A[k + 1], m + 1, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[0], m + 1, B, ldb, alpha, &B[k], ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                    k, n, ONE, &A[k], m + 1, &B[k], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                    k, n, alpha, &A[k], m + 1, &B[k], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[0], m + 1, &B[k], ldb, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                    k, n, ONE, &A[k + 1], m + 1, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                    k, n, alpha, &A[k], k, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[k * (k + 1)], k, B, ldb, alpha, &B[k], ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                    k, n, ONE, &A[0], k, &B[k], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                    k, n, alpha, &A[0], k, &B[k], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[k * (k + 1)], k, &B[k], ldb, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                    k, n, ONE, &A[k], k, B, ldb);

                    }

                } else {

                    if (!notrans) {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, cblas_siag,
                                    k, n, alpha, &A[k * (k + 1)], k, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[0], k, B, ldb, alpha, &B[k], ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, cblas_siag,
                                    k, n, ONE, &A[k * k], k, &B[k], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, cblas_siag,
                                    k, n, alpha, &A[k * k], k, &B[k], ldb);
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    k, n, k, -ONE, &A[0], k, &B[k], ldb, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, cblas_siag,
                                    k, n, ONE, &A[k * (k + 1)], k, B, ldb);

                    }

                }

            }

        }

    } else {

        if (n % 2 == 0) {
            nisodd = 0;
            k = n / 2;
        } else {
            nisodd = 1;
            if (lower) {
                n2 = n / 2;
                n1 = n - n2;
            } else {
                n1 = n / 2;
                n2 = n - n1;
            }
        }

        if (nisodd) {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, n2, alpha, &A[n], n, &B[n1 * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n1, n2, -ONE, &B[n1 * ldb], ldb, &A[n1], n, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, n1, ONE, &A[0], n, B, ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, n1, alpha, &A[0], n, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, n2, n1, -ONE, B, ldb, &A[n1], n, alpha, &B[n1 * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, n2, ONE, &A[n], n, &B[n1 * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, n1, alpha, &A[n2], n, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n2, n1, -ONE, B, ldb, &A[0], n, alpha, &B[n1 * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, n2, ONE, &A[n1], n, &B[n1 * ldb], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, n2, alpha, &A[n1], n, &B[n1 * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, n1, n2, -ONE, &B[n1 * ldb], ldb, &A[0], n, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, n1, ONE, &A[n2], n, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, n2, alpha, &A[1], n1, &B[n1 * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, n1, n2, -ONE, &B[n1 * ldb], ldb, &A[n1 * n1], n1, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, n1, ONE, &A[0], n1, B, ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, n1, alpha, &A[0], n1, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n2, n1, -ONE, B, ldb, &A[n1 * n1], n1, alpha, &B[n1 * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, n2, ONE, &A[1], n1, &B[n1 * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, n1, alpha, &A[n2 * n2], n2, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, n2, n1, -ONE, B, ldb, &A[0], n2, alpha, &B[n1 * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, n2, ONE, &A[n1 * n2], n2, &B[n1 * ldb], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, n2, alpha, &A[n1 * n2], n2, &B[n1 * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, n1, n2, -ONE, &B[n1 * ldb], ldb, &A[0], n2, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, n1, ONE, &A[n2 * n2], n2, B, ldb);

                    }

                }

            }

        } else {

            if (normaltransr) {

                if (lower) {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, k, alpha, &A[0], n + 1, &B[k * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, -ONE, &B[k * ldb], ldb, &A[k + 1], n + 1, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, k, ONE, &A[1], n + 1, B, ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, k, alpha, &A[1], n + 1, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, k, k, -ONE, B, ldb, &A[k + 1], n + 1, alpha, &B[k * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, k, ONE, &A[0], n + 1, &B[k * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, k, alpha, &A[k + 1], n + 1, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, -ONE, B, ldb, &A[0], n + 1, alpha, &B[k * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, k, ONE, &A[k], n + 1, &B[k * ldb], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, k, alpha, &A[k], n + 1, &B[k * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, k, k, -ONE, &B[k * ldb], ldb, &A[0], n + 1, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, k, ONE, &A[k + 1], n + 1, B, ldb);

                    }

                }

            } else {

                if (lower) {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, k, alpha, &A[0], k, &B[k * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, k, k, -ONE, &B[k * ldb], ldb, &A[(k + 1) * k], k, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, k, ONE, &A[k], k, B, ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, k, alpha, &A[k], k, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, -ONE, B, ldb, &A[(k + 1) * k], k, alpha, &B[k * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, k, ONE, &A[0], k, &B[k * ldb], ldb);

                    }

                } else {

                    if (notrans) {

                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, cblas_siag,
                                    m, k, alpha, &A[(k + 1) * k], k, B, ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    m, k, k, -ONE, B, ldb, &A[0], k, alpha, &B[k * ldb], ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, cblas_siag,
                                    m, k, ONE, &A[k * k], k, &B[k * ldb], ldb);

                    } else {

                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, cblas_siag,
                                    m, k, alpha, &A[k * k], k, &B[k * ldb], ldb);
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, k, k, -ONE, &B[k * ldb], ldb, &A[0], k, alpha, B, ldb);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, cblas_siag,
                                    m, k, ONE, &A[(k + 1) * k], k, B, ldb);

                    }

                }

            }

        }

    }
}
