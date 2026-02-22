/**
 * @file chfrk.c
 * @brief CHFRK performs a Hermitian rank-k operation for matrix in RFP format.
 */

#include "semicolon_lapack_complex_single.h"
#include "semicolon_cblas.h"
#include <complex.h>

/**
 * CHFRK performs one of the Hermitian rank-k operations
 *
 *    C := alpha*A*A**H + beta*C,
 *
 * or
 *
 *    C := alpha*A**H*A + beta*C,
 *
 * where alpha and beta are real scalars, C is an n-by-n Hermitian
 * matrix and A is an n-by-k matrix in the first case and a k-by-n
 * matrix in the second case.
 *
 * @param[in] transr
 *          = 'N':  The Normal Form of RFP A is stored;
 *          = 'C':  The Conjugate-transpose Form of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular part of C is stored;
 *          = 'L':  Lower triangular part of C is stored.
 *
 * @param[in] trans
 *          = 'N':  C := alpha*A*A**H + beta*C.
 *          = 'C':  C := alpha*A**H*A + beta*C.
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          With TRANS = 'N', K specifies the number of columns of A.
 *          With TRANS = 'C', K specifies the number of rows of A.
 *          k >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          Single complex array, dimension (lda, ka)
 *          where ka is k when TRANS = 'N', and is n otherwise.
 *
 * @param[in] lda
 *          The leading dimension of A.
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          Single complex array, dimension (n*(n+1)/2).
 *          On entry, the matrix A in RFP Format. RFP Format is
 *          described by TRANSR, UPLO and N. Note that the imaginary
 *          parts of the diagonal elements need not be set, they are
 *          assumed to be zero, and on exit they are set to zero.
 */
void chfrk(
    const char* transr,
    const char* uplo,
    const char* trans,
    const INT n,
    const INT k,
    const f32 alpha,
    const c64* restrict A,
    const INT lda,
    const f32 beta,
    c64* restrict C)
{
    INT lower, normaltransr, nisodd, notrans;
    INT nrowa, j, nk, n1, n2;
    INT info;
    c64 calpha, cbeta;

    info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    notrans = (trans[0] == 'N' || trans[0] == 'n');

    if (notrans) {
        nrowa = n;
    } else {
        nrowa = k;
    }

    if (!normaltransr && !(transr[0] == 'C' || transr[0] == 'c')) {
        info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        info = -2;
    } else if (!notrans && !(trans[0] == 'C' || trans[0] == 'c')) {
        info = -3;
    } else if (n < 0) {
        info = -4;
    } else if (k < 0) {
        info = -5;
    } else if (lda < (1 > nrowa ? 1 : nrowa)) {
        info = -8;
    }
    if (info != 0) {
        xerbla("CHFRK ", -info);
        return;
    }

    if ((n == 0) || (((alpha == 0.0f) || (k == 0)) && (beta == 1.0f))) {
        return;
    }

    if ((alpha == 0.0f) && (beta == 0.0f)) {
        for (j = 0; j < (n * (n + 1)) / 2; j++) {
            C[j] = CMPLXF(0.0f, 0.0f);
        }
        return;
    }

    calpha = CMPLXF(alpha, 0.0f);
    cbeta = CMPLXF(beta, 0.0f);

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

                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n2, k, alpha, A + n1, lda,
                                beta, C + n, n);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                n2, n1, k, &calpha, A + n1, lda,
                                A, lda, &cbeta, C + n1, n);

                } else {

                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                n2, k, alpha, A + n1 * lda, lda,
                                beta, C + n, n);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                n2, n1, k, &calpha, A + n1 * lda, lda,
                                A, lda, &cbeta, C + n1, n);

                }

            } else {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2, n);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n2, k, alpha, A + (n2 - 1), lda,
                                beta, C + n1, n);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                n1, n2, k, &calpha, A, lda,
                                A + (n2 - 1), lda, &cbeta, C, n);

                } else {

                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2, n);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                n2, k, alpha, A + (n2 - 1) * lda, lda,
                                beta, C + n1, n);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                n1, n2, k, &calpha, A, lda,
                                A + (n2 - 1) * lda, lda, &cbeta, C, n);

                }

            }

        } else {

            if (lower) {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n1);
                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                n2, k, alpha, A + n1, lda,
                                beta, C + 1, n1);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                n1, n2, k, &calpha, A, lda,
                                A + n1, lda, &cbeta, C + n1 * n1, n1);

                } else {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                n1, k, alpha, A, lda,
                                beta, C, n1);
                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                n2, k, alpha, A + n1 * lda, lda,
                                beta, C + 1, n1);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                n1, n2, k, &calpha, A, lda,
                                A + n1 * lda, lda, &cbeta, C + n1 * n1, n1);

                }

            } else {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2 * n2, n2);
                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                n2, k, alpha, A + n1, lda,
                                beta, C + n1 * n2, n2);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                n2, n1, k, &calpha, A + n1, lda,
                                A, lda, &cbeta, C, n2);

                } else {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                n1, k, alpha, A, lda,
                                beta, C + n2 * n2, n2);
                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                n2, k, alpha, A + n1 * lda, lda,
                                beta, C + n1 * n2, n2);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                n2, n1, k, &calpha, A + n1 * lda, lda,
                                A, lda, &cbeta, C, n2);

                }

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + 1, n + 1);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C, n + 1);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                nk, nk, k, &calpha, A + nk, lda,
                                A, lda, &cbeta, C + nk + 1, n + 1);

                } else {

                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                nk, k, alpha, A, lda,
                                beta, C + 1, n + 1);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C, n + 1);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                nk, nk, k, &calpha, A + nk * lda, lda,
                                A, lda, &cbeta, C + nk + 1, n + 1);

                }

            } else {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk + 1, n + 1);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C + nk, n + 1);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                nk, nk, k, &calpha, A, lda,
                                A + nk, lda, &cbeta, C, n + 1);

                } else {

                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk + 1, n + 1);
                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C + nk, n + 1);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                nk, nk, k, &calpha, A, lda,
                                A + nk * lda, lda, &cbeta, C, n + 1);

                }

            }

        } else {

            if (lower) {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk, nk);
                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C, nk);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                nk, nk, k, &calpha, A, lda,
                                A + nk, lda, &cbeta, C + (nk + 1) * nk, nk);

                } else {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk, nk);
                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C, nk);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                nk, nk, k, &calpha, A, lda,
                                A + nk * lda, lda, &cbeta, C + (nk + 1) * nk, nk);

                }

            } else {

                if (notrans) {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk * (nk + 1), nk);
                    cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                                nk, k, alpha, A + nk, lda,
                                beta, C + nk * nk, nk);
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                nk, nk, k, &calpha, A + nk, lda,
                                A, lda, &cbeta, C, nk);

                } else {

                    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                nk, k, alpha, A, lda,
                                beta, C + nk * (nk + 1), nk);
                    cblas_cherk(CblasColMajor, CblasLower, CblasConjTrans,
                                nk, k, alpha, A + nk * lda, lda,
                                beta, C + nk * nk, nk);
                    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                nk, nk, k, &calpha, A + nk * lda, lda,
                                A, lda, &cbeta, C, nk);

                }

            }

        }

    }
}
