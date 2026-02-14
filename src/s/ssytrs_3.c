/**
 * @file ssytrs_3.c
 * @brief SSYTRS_3 solves a system of linear equations A*X = B with a real symmetric matrix using the factorization computed by SSYTRF_RK or DSYTRF_BK.
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSYTRS_3 solves a system of linear equations A * X = B with a real
 * symmetric matrix A using the factorization computed
 * by SSYTRF_RK or DSYTRF_BK:
 *
 *    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This algorithm is using Level 3 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are
 *          stored as an upper or lower triangular matrix:
 *          = 'U':  Upper triangular, form is A = P*U*D*(U**T)*(P**T);
 *          = 'L':  Lower triangular, form is A = P*L*D*(L**T)*(P**T).
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in] A
 *          Double precision array, dimension (lda, n).
 *          Diagonal of the block diagonal matrix D and factors U or L
 *          as computed by SSYTRF_RK and DSYTRF_BK.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] E
 *          Double precision array, dimension (n).
 *          Contains the superdiagonal (or subdiagonal) elements of the
 *          symmetric block diagonal matrix D.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ssytrs_3(
    const char* uplo,
    const int n,
    const int nrhs,
    const f32* restrict A,
    const int lda,
    const f32* restrict E,
    const int* restrict ipiv,
    f32* restrict B,
    const int ldb,
    int* info)
{
    const f32 ONE = 1.0f;

    int upper;
    int i, j, k, kp;
    f32 ak, akm1, akm1k, bk, bkm1, denom;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("SSYTRS_3", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (upper) {

        for (k = n - 1; k >= 0; k--) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        i = n - 1;
        while (i >= 0) {
            if (ipiv[i] >= 0) {
                cblas_sscal(nrhs, ONE / A[i + i * lda], &B[i + 0 * ldb], ldb);
            } else if (i > 0) {
                akm1k = E[i];
                akm1 = A[i - 1 + (i - 1) * lda] / akm1k;
                ak = A[i + i * lda] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[i - 1 + j * ldb] / akm1k;
                    bk = B[i + j * ldb] / akm1k;
                    B[i - 1 + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[i + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                i = i - 1;
            }
            i = i - 1;
        }

        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        for (k = 0; k < n; k++) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

    } else {

        for (k = 0; k < n; k++) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        i = 0;
        while (i < n) {
            if (ipiv[i] >= 0) {
                cblas_sscal(nrhs, ONE / A[i + i * lda], &B[i + 0 * ldb], ldb);
            } else if (i < n - 1) {
                akm1k = E[i];
                akm1 = A[i + i * lda] / akm1k;
                ak = A[i + 1 + (i + 1) * lda] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[i + j * ldb] / akm1k;
                    bk = B[i + 1 + j * ldb] / akm1k;
                    B[i + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[i + 1 + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                i = i + 1;
            }
            i = i + 1;
        }

        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        for (k = n - 1; k >= 0; k--) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_sswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }
    }
}
