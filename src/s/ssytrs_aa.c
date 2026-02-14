/**
 * @file ssytrs_aa.c
 * @brief SSYTRS_AA solves a system of linear equations A*X = B with a real symmetric matrix using the factorization computed by SSYTRF_AA.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSYTRS_AA solves a system of linear equations A*X = B with a real
 * symmetric matrix A using the factorization A = U**T*T*U or
 * A = L*T*L**T computed by SSYTRF_AA.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U**T*T*U;
 *          = 'L':  Lower triangular, form is A = L*T*L**T.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B. nrhs >= 0.
 *
 * @param[in] A
 *          Double precision array, dimension (lda, n).
 *          Details of factors computed by SSYTRF_AA.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges as computed by SSYTRF_AA.
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] work
 *          Double precision array, dimension (max(1, lwork)).
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          If min(n, nrhs) = 0, lwork >= 1, else lwork >= 3*n-2.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void ssytrs_aa(
    const char* uplo,
    const int n,
    const int nrhs,
    const f32* restrict A,
    const int lda,
    const int* restrict ipiv,
    f32* restrict B,
    const int ldb,
    f32* restrict work,
    const int lwork,
    int* info)
{
    int upper, lquery;
    int k, kp, lwkmin;
    int minval;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    minval = (n < nrhs) ? n : nrhs;
    if (minval == 0) {
        lwkmin = 1;
    } else {
        lwkmin = 3 * n - 2;
    }

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (lwork < lwkmin && !lquery) {
        *info = -10;
    }

    if (*info != 0) {
        xerbla("SSYTRS_AA", -(*info));
        return;
    } else if (lquery) {
        work[0] = (f32)lwkmin;
        return;
    }

    if (minval == 0) {
        return;
    }

    if (upper) {

        if (n > 1) {

            for (k = 0; k < n; k++) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }

            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                        n - 1, nrhs, 1.0f, &A[0 + 1 * lda], lda, &B[1 + 0 * ldb], ldb);
        }

        for (k = 0; k < n; k++) {
            work[n - 1 + k] = A[k + k * lda];
        }
        if (n > 1) {
            for (k = 0; k < n - 1; k++) {
                work[k] = A[k + (k + 1) * lda];
                work[2 * n - 1 + k] = A[k + (k + 1) * lda];
            }
        }
        sgtsv(n, nrhs, &work[0], &work[n - 1], &work[2 * n - 1], B, ldb, info);

        if (n > 1) {

            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                        n - 1, nrhs, 1.0f, &A[0 + 1 * lda], lda, &B[1 + 0 * ldb], ldb);

            for (k = n - 1; k >= 0; k--) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }
        }

    } else {

        if (n > 1) {

            for (k = 0; k < n; k++) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }

            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        n - 1, nrhs, 1.0f, &A[1 + 0 * lda], lda, &B[1 + 0 * ldb], ldb);
        }

        for (k = 0; k < n; k++) {
            work[n - 1 + k] = A[k + k * lda];
        }
        if (n > 1) {
            for (k = 0; k < n - 1; k++) {
                work[k] = A[(k + 1) + k * lda];
                work[2 * n - 1 + k] = A[(k + 1) + k * lda];
            }
        }
        sgtsv(n, nrhs, &work[0], &work[n - 1], &work[2 * n - 1], B, ldb, info);

        if (n > 1) {

            cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                        n - 1, nrhs, 1.0f, &A[1 + 0 * lda], lda, &B[1 + 0 * ldb], ldb);

            for (k = n - 1; k >= 0; k--) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }
        }
    }
}
