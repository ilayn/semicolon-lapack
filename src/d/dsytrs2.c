/**
 * @file dsytrs2.c
 * @brief DSYTRS2 solves a system of linear equations A*X = B with a real
 *        symmetric matrix A using the factorization computed by DSYTRF
 *        and converted by DSYCONV.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSYTRS2 solves a system of linear equations A*X = B with a real
 * symmetric matrix A using the factorization A = U*D*U**T or
 * A = L*D*L**T computed by DSYTRF and converted by DSYCONV.
 *
 * @param[in]     uplo  Specifies whether the details of the factorization
 *                      are stored as an upper or lower triangular matrix.
 *                      = 'U': Upper triangular, form is A = U*D*U**T;
 *                      = 'L': Lower triangular, form is A = L*D*L**T.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      The block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L as computed by DSYTRF.
 *                      Note that A is input/output. At the start of the
 *                      subroutine, we permute A in a "better" form and then
 *                      permute A back to its original form at the end.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     ipiv  Integer array, dimension (n).
 *                      Details of the interchanges and the block structure of D
 *                      as determined by DSYTRF.
 * @param[in,out] B     Double precision array, dimension (ldb, nrhs).
 *                      On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    work  Double precision array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dsytrs2(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    f64* restrict B,
    const INT ldb,
    f64* restrict work,
    INT* info)
{
    const f64 ONE = 1.0;

    INT upper;
    INT i, iinfo, j, k, kp;
    f64 ak, akm1, akm1k, bk, bkm1, denom;

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
        *info = -8;
    }
    if (*info != 0) {
        xerbla("DSYTRS2", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    dsyconv(uplo, "C", n, A, lda, ipiv, work, &iinfo);

    if (upper) {

        /* Solve A*X = B, where A = U*D*U**T. */

        /* P**T * B */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k-1 and -ipiv[k]-1 (0-based). */
                kp = -ipiv[k] - 1;
                if (kp == -ipiv[k - 1] - 1) {
                    cblas_dswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }
                k = k - 2;
            }
        }

        /* Compute (U \ P**T * B) -> B    [ (U \ P**T * B) ] */
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasUnit, n, nrhs, ONE, A, lda, B, ldb);

        /* Compute D \ B -> B   [ D \ (U \ P**T * B) ] */
        i = n - 1;
        while (i >= 0) {
            if (ipiv[i] >= 0) {
                cblas_dscal(nrhs, ONE / A[i + i * lda], &B[i], ldb);
            } else if (i > 0) {
                if (ipiv[i - 1] == ipiv[i]) {
                    akm1k = work[i];
                    akm1 = A[(i - 1) + (i - 1) * lda] / akm1k;
                    ak = A[i + i * lda] / akm1k;
                    denom = akm1 * ak - ONE;
                    for (j = 0; j < nrhs; j++) {
                        bkm1 = B[(i - 1) + j * ldb] / akm1k;
                        bk = B[i + j * ldb] / akm1k;
                        B[(i - 1) + j * ldb] = (ak * bkm1 - bk) / denom;
                        B[i + j * ldb] = (akm1 * bk - bkm1) / denom;
                    }
                    i = i - 1;
                }
            }
            i = i - 1;
        }

        /* Compute (U**T \ B) -> B   [ U**T \ (D \ (U \ P**T * B)) ] */
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                    CblasUnit, n, nrhs, ONE, A, lda, B, ldb);

        /* P * B  [ P * (U**T \ (D \ (U \ P**T * B))) ] */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k and -ipiv[k]-1 (0-based). */
                kp = -ipiv[k] - 1;
                if (k < n - 1 && kp == -ipiv[k + 1] - 1) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 2;
            }
        }

    } else {

        /* Solve A*X = B, where A = L*D*L**T. */

        /* P**T * B */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k+1 and -ipiv[k+1]-1 (0-based). */
                kp = -ipiv[k + 1] - 1;
                if (kp == -ipiv[k] - 1) {
                    cblas_dswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }
                k = k + 2;
            }
        }

        /* Compute (L \ P**T * B) -> B    [ (L \ P**T * B) ] */
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasUnit, n, nrhs, ONE, A, lda, B, ldb);

        /* Compute D \ B -> B   [ D \ (L \ P**T * B) ] */
        i = 0;
        while (i < n) {
            if (ipiv[i] >= 0) {
                cblas_dscal(nrhs, ONE / A[i + i * lda], &B[i], ldb);
            } else {
                akm1k = work[i];
                akm1 = A[i + i * lda] / akm1k;
                ak = A[(i + 1) + (i + 1) * lda] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[i + j * ldb] / akm1k;
                    bk = B[(i + 1) + j * ldb] / akm1k;
                    B[i + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[(i + 1) + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                i = i + 1;
            }
            i = i + 1;
        }

        /* Compute (L**T \ B) -> B   [ L**T \ (D \ (L \ P**T * B)) ] */
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                    CblasUnit, n, nrhs, ONE, A, lda, B, ldb);

        /* P * B  [ P * (L**T \ (D \ (L \ P**T * B))) ] */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k and -ipiv[k]-1 (0-based). */
                kp = -ipiv[k] - 1;
                if (k > 0 && kp == -ipiv[k - 1] - 1) {
                    cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 2;
            }
        }

    }

    dsyconv(uplo, "R", n, A, lda, ipiv, work, &iinfo);
}
