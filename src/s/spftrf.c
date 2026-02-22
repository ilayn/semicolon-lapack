/**
 * @file spftrf.c
 * @brief SPFTRF computes the Cholesky factorization of a symmetric positive definite matrix in RFP format.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"

/**
 * SPFTRF computes the Cholesky factorization of a real symmetric
 * positive definite matrix A.
 *
 * The factorization has the form
 *    A = U**T * U,  if UPLO = 'U', or
 *    A = L  * L**T,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the block version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] transr
 *          = 'N':  The Normal TRANSR of RFP A is stored;
 *          = 'T':  The Transpose TRANSR of RFP A is stored.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of RFP A is stored;
 *          = 'L':  Lower triangle of RFP A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (n*(n+1)/2).
 *          On entry, the symmetric matrix A in RFP format.
 *          On exit, the factor U or L from the Cholesky factorization.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the leading principal minor of order i
 *                           is not positive, and the factorization could not be
 *                           completed.
 */
void spftrf(
    const char* transr,
    const char* uplo,
    const INT n,
    f32* restrict A,
    INT* info)
{
    INT lower, nisodd, normaltransr;
    INT n1, n2, k;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    if (!normaltransr && !(transr[0] == 'T' || transr[0] == 't')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("SPFTRF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (n % 2 == 0) {
        k = n / 2;
        nisodd = 0;
    } else {
        nisodd = 1;
    }

    if (lower) {
        n2 = n / 2;
        n1 = n - n2;
    } else {
        n1 = n / 2;
        n2 = n - n1;
    }

    if (nisodd) {

        if (normaltransr) {

            if (lower) {

                spotrf("L", n1, A, n, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans, CblasNonUnit,
                            n2, n1, 1.0f, A, n, &A[n1], n);
                cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                            n2, n1, -1.0f, &A[n1], n, 1.0f, &A[n], n);
                spotrf("U", n2, &A[n], n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            } else {

                spotrf("L", n1, &A[n2], n, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            n1, n2, 1.0f, &A[n2], n, A, n);
                cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                            n2, n1, -1.0f, A, n, 1.0f, &A[n1], n);
                spotrf("U", n2, &A[n1], n, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            }

        } else {

            if (lower) {

                spotrf("U", n1, A, n1, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasTrans, CblasNonUnit,
                            n1, n2, 1.0f, A, n1, &A[n1 * n1], n1);
                cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans,
                            n2, n1, -1.0f, &A[n1 * n1], n1, 1.0f, &A[1], n1);
                spotrf("L", n2, &A[1], n1, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            } else {

                spotrf("U", n1, &A[n2 * n2], n2, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            n2, n1, 1.0f, &A[n2 * n2], n2, A, n2);
                cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            n2, n1, -1.0f, A, n2, 1.0f, &A[n1 * n2], n2);
                spotrf("L", n2, &A[n1 * n2], n2, info);
                if (*info > 0) {
                    *info = *info + n1;
                }

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                spotrf("L", k, &A[1], n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasRight, CblasLower,
                            CblasTrans, CblasNonUnit,
                            k, k, 1.0f, &A[1], n + 1, &A[k + 1], n + 1);
                cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                            k, k, -1.0f, &A[k + 1], n + 1, 1.0f, A, n + 1);
                spotrf("U", k, A, n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            } else {

                spotrf("L", k, &A[k + 1], n + 1, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasLeft, CblasLower,
                            CblasNoTrans, CblasNonUnit,
                            k, k, 1.0f, &A[k + 1], n + 1, A, n + 1);
                cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                            k, k, -1.0f, A, n + 1, 1.0f, &A[k], n + 1);
                spotrf("U", k, &A[k], n + 1, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            }

        } else {

            if (lower) {

                spotrf("U", k, &A[k], k, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasLeft, CblasUpper,
                            CblasTrans, CblasNonUnit,
                            k, k, 1.0f, &A[k], k, &A[k * (k + 1)], k);
                cblas_ssyrk(CblasColMajor, CblasLower, CblasTrans,
                            k, k, -1.0f, &A[k * (k + 1)], k, 1.0f, A, k);
                spotrf("L", k, A, k, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            } else {

                spotrf("U", k, &A[k * (k + 1)], k, info);
                if (*info > 0) {
                    return;
                }
                cblas_strsm(CblasColMajor, CblasRight, CblasUpper,
                            CblasNoTrans, CblasNonUnit,
                            k, k, 1.0f, &A[k * (k + 1)], k, A, k);
                cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            k, k, -1.0f, A, k, 1.0f, &A[k * k], k);
                spotrf("L", k, &A[k * k], k, info);
                if (*info > 0) {
                    *info = *info + k;
                }

            }

        }

    }
}
