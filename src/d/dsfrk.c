/**
 * @file dsfrk.c
 * @brief DSFRK performs a symmetric rank-k operation for matrix in RFP format.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>

/**
 * DSFRK performs one of the symmetric rank-k operations
 *
 *    C := alpha*A*A**T + beta*C,
 *
 * or
 *
 *    C := alpha*A**T*A + beta*C,
 *
 * where alpha and beta are real scalars, C is an n-by-n symmetric
 * matrix and A is an n-by-k matrix in the first case and a k-by-n
 * matrix in the second case.
 *
 * @param[in] transr
 *          = 'N':  The Normal Form of RFP A is stored;
 *          = 'T':  The Transpose Form of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular part of C is stored;
 *          = 'L':  Lower triangular part of C is stored.
 *
 * @param[in] trans
 *          = 'N':  C := alpha*A*A**T + beta*C.
 *          = 'T':  C := alpha*A**T*A + beta*C.
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          With TRANS = 'N', K specifies the number of columns of A.
 *          With TRANS = 'T', K specifies the number of rows of A.
 *          k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Double precision array, dimension (lda, ka)
 *          where ka is k when TRANS = 'N', and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of A.
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          Double precision array, dimension (n*(n+1)/2).
 *          The symmetric matrix C in RFP format.
 */
void dsfrk(
    const char* transr,
    const char* uplo,
    const char* trans,
    const int n,
    const int k,
    const f64 alpha,
    const f64* restrict A,
    const int lda,
    const f64 beta,
    f64* restrict C)
{
    int lower, normaltransr, nisodd, notrans;
    int nrowa, j, nk, n1, n2;
    int info;

    info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    notrans = (trans[0] == 'N' || trans[0] == 'n');

    if (notrans) {
        nrowa = n;
    } else {
        nrowa = k;
    }

    if (!normaltransr && !(transr[0] == 'T' || transr[0] == 't')) {
        info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        info = -2;
    } else if (!notrans && !(trans[0] == 'T' || trans[0] == 't')) {
        info = -3;
    } else if (n < 0) {
        info = -4;
    } else if (k < 0) {
        info = -5;
    } else if (lda < (1 > nrowa ? 1 : nrowa)) {
        info = -8;
    }
    if (info != 0) {
        xerbla("DSFRK ", -info);
        return;
    }

    if ((n == 0) || (((alpha == 0.0) || (k == 0)) && (beta == 1.0))) {
        return;
    }

    if ((alpha == 0.0) && (beta == 0.0)) {
        for (j = 0; j < (n * (n + 1)) / 2; j++) {
            C[j] = 0.0;
        }
        return;
    }

    if (n % 2 == 0) {
        nisodd = 0;
        nk = n / 2;
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

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n2, k, alpha, A + n1, lda,
                                beta, C + n, n);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n2, n1, k, alpha, A + n1, lda,
                                A, lda, beta, C + n1, n);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                n2, k, alpha, A + n1 * lda, lda,
                                beta, C + n, n);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                n2, n1, k, alpha, A + n1 * lda, lda,
                                A, lda, beta, C + n1, n);

                }

            } else {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2, n);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n2, k, alpha, A + (n2 - 1), lda,
                                beta, C + n1, n);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n2, k, alpha, A, lda,
                                A + (n2 - 1), lda, beta, C, n);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2, n);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                n2, k, alpha, A + (n2 - 1) * lda, lda,
                                beta, C + n1, n);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                n1, n2, k, alpha, A, lda,
                                A + (n2 - 1) * lda, lda, beta, C, n);

                }

            }

        } else {

            if (lower) {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n1);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                n2, k, alpha, A + n1, lda,
                                beta, C + 1, n1);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n2, k, alpha, A, lda,
                                A + n1, lda, beta, C + n1 * n1, n1);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n1);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                n2, k, alpha, A + n1 * lda, lda,
                                beta, C + 1, n1);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                n1, n2, k, alpha, A, lda,
                                A + n1 * lda, lda, beta, C + n1 * n1, n1);

                }

            } else {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2 * n2, n2);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                n2, k, alpha, A + n1, lda,
                                beta, C + n1 * n2, n2);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n2, n1, k, alpha, A + n1, lda,
                                A, lda, beta, C, n2);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2 * n2, n2);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                n2, k, alpha, A + n1 * lda, lda,
                                beta, C + n1 * n2, n2);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                n2, n1, k, alpha, A + n1 * lda, lda,
                                A, lda, beta, C, n2);

                }

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + 1, n + 1);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C, n + 1);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                nk, nk, k, alpha, A + nk, lda,
                                A, lda, beta, C + nk + 1, n + 1);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                nk, k, alpha, A, lda,
                                beta, C + 1, n + 1);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C, n + 1);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                nk, nk, k, alpha, A + nk * lda, lda,
                                A, lda, beta, C + nk + 1, n + 1);

                }

            } else {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk + 1, n + 1);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C + nk, n + 1);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                nk, nk, k, alpha, A, lda,
                                A + nk, lda, beta, C, n + 1);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk + 1, n + 1);
                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C + nk, n + 1);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                nk, nk, k, alpha, A, lda,
                                A + nk * lda, lda, beta, C, n + 1);

                }

            }

        } else {

            if (lower) {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk, nk);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C, nk);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                nk, nk, k, alpha, A, lda,
                                A + nk, lda, beta, C + (nk + 1) * nk, nk);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk, nk);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C, nk);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                nk, nk, k, alpha, A, lda,
                                A + nk * lda, lda, beta, C + (nk + 1) * nk, nk);

                }

            } else {

                if (notrans) {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk * (nk + 1), nk);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C + nk * nk, nk);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                nk, nk, k, alpha, A + nk, lda,
                                A, lda, beta, C, nk);

                } else {

                    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk * (nk + 1), nk);
                    cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C + nk * nk, nk);
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                nk, nk, k, alpha, A + nk * lda, lda,
                                A, lda, beta, C, nk);

                }

            }

        }

    }
}
