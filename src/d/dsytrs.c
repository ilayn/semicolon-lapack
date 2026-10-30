/**
 * @file dsytrs.c
 * @brief DSYTRS solves a system of linear equations A*X = B with a
 *        real symmetric matrix A using the factorization computed by DSYTRF.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSYTRS solves a system of linear equations A*X = B with a real
 * symmetric matrix A using the factorization A = U*D*U**T or
 * A = L*D*L**T computed by DSYTRF.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangular, form is A = U*D*U**T
 *                      - `'L'`: Lower triangular, form is A = L*D*L**T
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B. `nrhs>=0`.
 * @param[in]     A     Array of dimension `(lda,n)`.
 *                      The block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L as computed by `dsytrf`.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[in]     ipiv  Array of dimension `n`.
 *                      Details of the interchanges and the block structure
 *                      of D as determined by `dsytrf`.
 * @param[in,out] B     Array of dimension `(ldb,nrhs)`.
 *                      On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 * @param[in]     ldb   The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void dsytrs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("DSYTRS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) return;

    if (upper) {

        /* Solve A*X = B, where A = U*D*U**T.
         *
         * First solve U*D*X = B, overwriting B with X.
         *
         * K is the main loop index, decreasing from n-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        INT k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 *
                 * Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(U(k)), where U(k) is the transformation
                 * stored in column k of A. */
                if (k > 0) {
                    cblas_dger(CblasColMajor, k, nrhs,
                               -1.0, &A[0 + k * lda], 1, &B[k], ldb,
                               &B[0], ldb);
                }

                /* Multiply by the inverse of the diagonal block. */
                cblas_dscal(nrhs, 1.0 / A[k + k * lda], &B[k], ldb);
                k--;
            } else {
                /* 2 x 2 diagonal block
                 *
                 * Interchange rows k-1 and -ipiv[k]-1. */
                INT kp = -(ipiv[k] + 1);
                if (kp != k - 1) {
                    cblas_dswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(U(k)), where U(k) is the transformation
                 * stored in columns k-1 and k of A. */
                if (k > 1) {
                    cblas_dger(CblasColMajor, k - 1, nrhs,
                               -1.0, &A[0 + k * lda], 1, &B[k], ldb,
                               &B[0], ldb);
                    cblas_dger(CblasColMajor, k - 1, nrhs,
                               -1.0, &A[0 + (k - 1) * lda], 1, &B[k - 1], ldb,
                               &B[0], ldb);
                }

                /* Multiply by the inverse of the diagonal block. */
                f64 akm1k = A[(k - 1) + k * lda];
                f64 akm1 = A[(k - 1) + (k - 1) * lda] / akm1k;
                f64 ak = A[k + k * lda] / akm1k;
                f64 denom = akm1 * ak - 1.0;
                for (INT j = 0; j < nrhs; j++) {
                    f64 bkm1 = B[(k - 1) + j * ldb] / akm1k;
                    f64 bk = B[k + j * ldb] / akm1k;
                    B[(k - 1) + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[k + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k -= 2;
            }
        }

        /* Next solve U**T *X = B, overwriting B with X.
         *
         * K is the main loop index, increasing from 0 to n-1 in steps of
         * 1 or 2, depending on the size of the diagonal blocks. */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 *
                 * Multiply by inv(U**T(k)), where U(k) is the transformation
                 * stored in column k of A. */
                if (k > 0) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                k, nrhs,
                                -1.0, &B[0], ldb, &A[0 + k * lda], 1,
                                1.0, &B[k], ldb);
                }

                /* Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k++;
            } else {
                /* 2 x 2 diagonal block
                 *
                 * Multiply by inv(U**T(k+1)), where U(k+1) is the transformation
                 * stored in columns k and k+1 of A. */
                if (k > 0) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                k, nrhs,
                                -1.0, &B[0], ldb, &A[0 + k * lda], 1,
                                1.0, &B[k], ldb);
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                k, nrhs,
                                -1.0, &B[0], ldb, &A[0 + (k + 1) * lda], 1,
                                1.0, &B[k + 1], ldb);
                }

                /* Interchange rows k and -ipiv[k]-1. */
                INT kp = -(ipiv[k] + 1);
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k += 2;
            }
        }

    } else {

        /* Solve A*X = B, where A = L*D*L**T.
         *
         * First solve L*D*X = B, overwriting B with X.
         *
         * K is the main loop index, increasing from 0 to n-1 in steps of
         * 1 or 2, depending on the size of the diagonal blocks. */
        INT k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 *
                 * Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(L(k)), where L(k) is the transformation
                 * stored in column k of A. */
                if (k < n - 1) {
                    cblas_dger(CblasColMajor, n - k - 1, nrhs,
                               -1.0, &A[(k + 1) + k * lda], 1, &B[k], ldb,
                               &B[k + 1], ldb);
                }

                /* Multiply by the inverse of the diagonal block. */
                cblas_dscal(nrhs, 1.0 / A[k + k * lda], &B[k], ldb);
                k++;
            } else {
                /* 2 x 2 diagonal block
                 *
                 * Interchange rows k+1 and -ipiv[k]-1. */
                INT kp = -(ipiv[k] + 1);
                if (kp != k + 1) {
                    cblas_dswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(L(k)), where L(k) is the transformation
                 * stored in columns k and k+1 of A. */
                if (k < n - 2) {
                    cblas_dger(CblasColMajor, n - k - 2, nrhs,
                               -1.0, &A[(k + 2) + k * lda], 1, &B[k], ldb,
                               &B[k + 2], ldb);
                    cblas_dger(CblasColMajor, n - k - 2, nrhs,
                               -1.0, &A[(k + 2) + (k + 1) * lda], 1, &B[k + 1], ldb,
                               &B[k + 2], ldb);
                }

                /* Multiply by the inverse of the diagonal block. */
                f64 akm1k = A[(k + 1) + k * lda];
                f64 akm1 = A[k + k * lda] / akm1k;
                f64 ak = A[(k + 1) + (k + 1) * lda] / akm1k;
                f64 denom = akm1 * ak - 1.0;
                for (INT j = 0; j < nrhs; j++) {
                    f64 bkm1 = B[k + j * ldb] / akm1k;
                    f64 bk = B[(k + 1) + j * ldb] / akm1k;
                    B[k + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[(k + 1) + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k += 2;
            }
        }

        /* Next solve L**T *X = B, overwriting B with X.
         *
         * K is the main loop index, decreasing from n-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks. */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 *
                 * Multiply by inv(L**T(k)), where L(k) is the transformation
                 * stored in column k of A. */
                if (k < n - 1) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                n - k - 1, nrhs,
                                -1.0, &B[k + 1], ldb, &A[(k + 1) + k * lda], 1,
                                1.0, &B[k], ldb);
                }

                /* Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k--;
            } else {
                /* 2 x 2 diagonal block
                 *
                 * Multiply by inv(L**T(k-1)), where L(k-1) is the transformation
                 * stored in columns k-1 and k of A. */
                if (k < n - 1) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                n - k - 1, nrhs,
                                -1.0, &B[k + 1], ldb, &A[(k + 1) + k * lda], 1,
                                1.0, &B[k], ldb);
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                n - k - 1, nrhs,
                                -1.0, &B[k + 1], ldb, &A[(k + 1) + (k - 1) * lda], 1,
                                1.0, &B[k - 1], ldb);
                }

                /* Interchange rows k and -ipiv[k]-1. */
                INT kp = -(ipiv[k] + 1);
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k -= 2;
            }
        }
    }
}
