/**
 * @file zsytrs_3.c
 * @brief ZSYTRS_3 solves a system of linear equations A*X = B with a complex symmetric matrix using the factorization computed by ZSYTRF_RK or ZSYTRF_BK.
 */

#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZSYTRS_3 solves a system of linear equations A * X = B with a complex
 * symmetric matrix A using the factorization computed
 * by ZSYTRF_RK or ZSYTRF_BK:
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
 *          Double complex array, dimension (lda, n).
 *          Diagonal of the block diagonal matrix D and factors U or L
 *          as computed by ZSYTRF_RK and ZSYTRF_BK.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] E
 *          Double complex array, dimension (n).
 *          Contains the superdiagonal (or subdiagonal) elements of the
 *          symmetric block diagonal matrix D.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
 *
 * @param[in,out] B
 *          Double complex array, dimension (ldb, nrhs).
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
void zsytrs_3(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* restrict A,
    const INT lda,
    const c128* restrict E,
    const INT* restrict ipiv,
    c128* restrict B,
    const INT ldb,
    INT* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);

    INT upper;
    INT i, j, k, kp;
    c128 ak, akm1, akm1k, bk, bkm1, denom;

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
        xerbla("ZSYTRS_3", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (upper) {

        for (k = n - 1; k >= 0; k--) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_zswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        i = n - 1;
        while (i >= 0) {
            if (ipiv[i] >= 0) {
                c128 scal = ONE / A[i + i * lda];
                cblas_zscal(nrhs, &scal, &B[i + 0 * ldb], ldb);
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

        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        for (k = 0; k < n; k++) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_zswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

    } else {

        for (k = 0; k < n; k++) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_zswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        i = 0;
        while (i < n) {
            if (ipiv[i] >= 0) {
                c128 scal = ONE / A[i + i * lda];
                cblas_zscal(nrhs, &scal, &B[i + 0 * ldb], ldb);
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

        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        for (k = n - 1; k >= 0; k--) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_zswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }
    }
}
