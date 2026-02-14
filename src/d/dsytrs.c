/**
 * @file dsytrs.c
 * @brief DSYTRS solves a system of linear equations A*X = B with a
 *        real symmetric matrix A using the factorization computed by DSYTRF.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSYTRS solves a system of linear equations A*X = B with a real
 * symmetric matrix A using the factorization A = U*D*U**T or
 * A = L*D*L**T computed by DSYTRF.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**T
 *                        = 'L': Lower triangular, A = L*D*L**T
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     A     Double precision array, dimension (lda, n).
 *                      The block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L as computed by DSYTRF.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from DSYTRF.
 * @param[in,out] B     Double precision array, dimension (ldb, nrhs).
 *                      On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dsytrs(
    const char* uplo,
    const int n,
    const int nrhs,
    const f64* const restrict A,
    const int lda,
    const int* const restrict ipiv,
    f64* const restrict B,
    const int ldb,
    int* info)
{
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
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
        int k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Interchange rows k and ipiv[k]. */
                int kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(U(k)), where U(k) is stored in column k.
                 * Fortran: DGER(K-1, NRHS, -1, A(1,K), 1, B(K,1), LDB, B(1,1), LDB)
                 * 0-based: M = k, rows 0..k-1 */
                if (k > 0) {
                    cblas_dger(CblasColMajor, k, nrhs,
                               -1.0, &A[0 + k * lda], 1, &B[k], ldb,
                               &B[0], ldb);
                }

                /* Multiply by inverse of diagonal block.
                 * Fortran: DSCAL(NRHS, 1/A(K,K), B(K,1), LDB) */
                cblas_dscal(nrhs, 1.0 / A[k + k * lda], &B[k], ldb);
                k--;
            } else {
                /* 2x2 diagonal block.
                 * Interchange rows k-1 and kp = -(ipiv[k]+1). */
                int kp = -(ipiv[k] + 1);
                if (kp != k - 1) {
                    cblas_dswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(U(k)), stored in columns k-1 and k.
                 * Fortran: DGER(K-2, NRHS, -1, A(1,K), ...) and DGER(K-2, ..., A(1,K-1), ...)
                 * 0-based: M = k-1, rows 0..k-2 */
                if (k > 1) {
                    cblas_dger(CblasColMajor, k - 1, nrhs,
                               -1.0, &A[0 + k * lda], 1, &B[k], ldb,
                               &B[0], ldb);
                    cblas_dger(CblasColMajor, k - 1, nrhs,
                               -1.0, &A[0 + (k - 1) * lda], 1, &B[k - 1], ldb,
                               &B[0], ldb);
                }

                /* Multiply by inverse of 2x2 diagonal block.
                 * D = [ A(k-1,k-1)  A(k-1,k) ]
                 *     [ A(k-1,k)    A(k,k)   ] */
                f64 akm1k = A[(k - 1) + k * lda];
                f64 akm1 = A[(k - 1) + (k - 1) * lda] / akm1k;
                f64 ak = A[k + k * lda] / akm1k;
                f64 denom = akm1 * ak - 1.0;
                for (int j = 0; j < nrhs; j++) {
                    f64 bkm1 = B[(k - 1) + j * ldb] / akm1k;
                    f64 bk = B[k + j * ldb] / akm1k;
                    B[(k - 1) + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[k + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k -= 2;
            }
        }

        /* Next solve U**T * X = B, overwriting B with X.
         *
         * K increases from 0 to n-1 in steps of 1 or 2. */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Multiply by inv(U**T(k)).
                 * Fortran: DGEMV('T', K-1, NRHS, -1, B, LDB, A(1,K), 1, 1, B(K,1), LDB)
                 * 0-based: M = k, source = B(0:k-1, :), vector = A(0:k-1, k) */
                if (k > 0) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                k, nrhs,
                                -1.0, &B[0], ldb, &A[0 + k * lda], 1,
                                1.0, &B[k], ldb);
                }

                /* Interchange rows k and ipiv[k]. */
                int kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k++;
            } else {
                /* 2x2 diagonal block.
                 * Multiply by inv(U**T(k+1)).
                 * Fortran: DGEMV('T', K-1, ...) on columns K and K+1
                 * 0-based: M = k */
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

                /* Interchange rows k and kp = -(ipiv[k]+1). */
                int kp = -(ipiv[k] + 1);
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
         * K increases from 0 to n-1 in steps of 1 or 2. */
        int k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Interchange rows k and ipiv[k]. */
                int kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(L(k)).
                 * Fortran: DGER(N-K, NRHS, -1, A(K+1,K), 1, B(K,1), LDB, B(K+1,1), LDB)
                 * 0-based: M = n-k-1, rows k+1..n-1 */
                if (k < n - 1) {
                    cblas_dger(CblasColMajor, n - k - 1, nrhs,
                               -1.0, &A[(k + 1) + k * lda], 1, &B[k], ldb,
                               &B[k + 1], ldb);
                }

                /* Multiply by inverse of diagonal block. */
                cblas_dscal(nrhs, 1.0 / A[k + k * lda], &B[k], ldb);
                k++;
            } else {
                /* 2x2 diagonal block.
                 * Interchange rows k+1 and kp = -(ipiv[k]+1). */
                int kp = -(ipiv[k] + 1);
                if (kp != k + 1) {
                    cblas_dswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(L(k)).
                 * Fortran: DGER(N-K-1, ..., A(K+2,K), ...) and DGER(..., A(K+2,K+1), ...)
                 * 0-based: M = n-k-2, rows k+2..n-1 */
                if (k < n - 2) {
                    cblas_dger(CblasColMajor, n - k - 2, nrhs,
                               -1.0, &A[(k + 2) + k * lda], 1, &B[k], ldb,
                               &B[k + 2], ldb);
                    cblas_dger(CblasColMajor, n - k - 2, nrhs,
                               -1.0, &A[(k + 2) + (k + 1) * lda], 1, &B[k + 1], ldb,
                               &B[k + 2], ldb);
                }

                /* Multiply by inverse of 2x2 diagonal block.
                 * D = [ A(k,k)    A(k+1,k) ]
                 *     [ A(k+1,k)  A(k+1,k+1) ] */
                f64 akm1k = A[(k + 1) + k * lda];
                f64 akm1 = A[k + k * lda] / akm1k;
                f64 ak = A[(k + 1) + (k + 1) * lda] / akm1k;
                f64 denom = akm1 * ak - 1.0;
                for (int j = 0; j < nrhs; j++) {
                    f64 bkm1 = B[k + j * ldb] / akm1k;
                    f64 bk = B[(k + 1) + j * ldb] / akm1k;
                    B[k + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[(k + 1) + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k += 2;
            }
        }

        /* Next solve L**T * X = B, overwriting B with X.
         *
         * K decreases from n-1 to 0 in steps of 1 or 2. */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Multiply by inv(L**T(k)).
                 * Fortran: DGEMV('T', N-K, NRHS, -1, B(K+1,1), LDB, A(K+1,K), 1, 1, B(K,1), LDB)
                 * 0-based: M = n-k-1, source = B(k+1:n-1, :), vector = A(k+1:n-1, k) */
                if (k < n - 1) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                n - k - 1, nrhs,
                                -1.0, &B[k + 1], ldb, &A[(k + 1) + k * lda], 1,
                                1.0, &B[k], ldb);
                }

                /* Interchange rows k and ipiv[k]. */
                int kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k--;
            } else {
                /* 2x2 diagonal block.
                 * Multiply by inv(L**T(k-1)).
                 * Fortran: DGEMV('T', N-K, ...) on columns K and K-1
                 * 0-based: M = n-k-1 */
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

                /* Interchange rows k and kp = -(ipiv[k]+1). */
                int kp = -(ipiv[k] + 1);
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k -= 2;
            }
        }
    }
}
