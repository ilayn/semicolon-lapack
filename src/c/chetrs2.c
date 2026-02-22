/**
 * @file chetrs2.c
 * @brief CHETRS2 solves a system of linear equations A*X = B with a complex
 *        Hermitian matrix A using the factorization computed by CHETRF
 *        and converted by CSYCONV.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRS2 solves a system of linear equations A*X = B with a complex
 * Hermitian matrix A using the factorization A = U*D*U**H or
 * A = L*D*L**H computed by CHETRF and converted by CSYCONV.
 *
 * @param[in]     uplo  Specifies whether the details of the factorization
 *                      are stored as an upper or lower triangular matrix.
 *                      = 'U': Upper triangular, form is A = U*D*U**H;
 *                      = 'L': Lower triangular, form is A = L*D*L**H.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      The block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L as computed by CHETRF.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     ipiv  Integer array, dimension (n).
 *                      Details of the interchanges and the block structure of D
 *                      as determined by CHETRF.
 * @param[in,out] B     Complex*16 array, dimension (ldb, nrhs).
 *                      On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    work  Complex*16 array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void chetrs2(
    const char* uplo,
    const INT n,
    const INT nrhs,
    c64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c64* restrict B,
    const INT ldb,
    c64* restrict work,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    INT upper;
    INT i, iinfo, j, k, kp;
    f32 s;
    c64 ak, akm1, akm1k, bk, bkm1, denom;

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
        xerbla("CHETRS2", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    csyconv(uplo, "C", n, A, lda, ipiv, work, &iinfo);

    if (upper) {

        /* Solve A*X = B, where A = U*D*U**H. */

        /* P**T * B */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k-1 and -ipiv[k]-1 (0-based). */
                kp = -ipiv[k] - 1;
                if (kp == -ipiv[k - 1] - 1) {
                    cblas_cswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }
                k = k - 2;
            }
        }

        /* Compute (U \P**T * B) -> B    [ (U \P**T * B) ] */
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasUnit, n, nrhs, &ONE, A, lda, B, ldb);

        /* Compute D \ B -> B   [ D \ (U \P**T * B) ] */
        i = n - 1;
        while (i >= 0) {
            if (ipiv[i] >= 0) {
                s = 1.0f / crealf(A[i + i * lda]);
                cblas_csscal(nrhs, s, &B[i], ldb);
            } else if (i > 0) {
                if (ipiv[i - 1] == ipiv[i]) {
                    akm1k = work[i];
                    akm1 = A[(i - 1) + (i - 1) * lda] / akm1k;
                    ak = A[i + i * lda] / conjf(akm1k);
                    denom = akm1 * ak - ONE;
                    for (j = 0; j < nrhs; j++) {
                        bkm1 = B[(i - 1) + j * ldb] / akm1k;
                        bk = B[i + j * ldb] / conjf(akm1k);
                        B[(i - 1) + j * ldb] = (ak * bkm1 - bk) / denom;
                        B[i + j * ldb] = (akm1 * bk - bkm1) / denom;
                    }
                    i = i - 1;
                }
            }
            i = i - 1;
        }

        /* Compute (U**H \ B) -> B   [ U**H \ (D \ (U \P**T * B)) ] */
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasUnit, n, nrhs, &ONE, A, lda, B, ldb);

        /* P * B  [ P * (U**H \ (D \ (U \P**T * B))) ] */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k and -ipiv[k]-1 (0-based). */
                kp = -ipiv[k] - 1;
                if (k < n - 1 && kp == -ipiv[k + 1] - 1) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 2;
            }
        }

    } else {

        /* Solve A*X = B, where A = L*D*L**H. */

        /* P**T * B */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k+1 and -ipiv[k+1]-1 (0-based). */
                kp = -ipiv[k + 1] - 1;
                if (kp == -ipiv[k] - 1) {
                    cblas_cswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }
                k = k + 2;
            }
        }

        /* Compute (L \P**T * B) -> B    [ (L \P**T * B) ] */
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasUnit, n, nrhs, &ONE, A, lda, B, ldb);

        /* Compute D \ B -> B   [ D \ (L \P**T * B) ] */
        i = 0;
        while (i < n) {
            if (ipiv[i] >= 0) {
                s = 1.0f / crealf(A[i + i * lda]);
                cblas_csscal(nrhs, s, &B[i], ldb);
            } else {
                akm1k = work[i];
                akm1 = A[i + i * lda] / conjf(akm1k);
                ak = A[(i + 1) + (i + 1) * lda] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[i + j * ldb] / conjf(akm1k);
                    bk = B[(i + 1) + j * ldb] / akm1k;
                    B[i + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[(i + 1) + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                i = i + 1;
            }
            i = i + 1;
        }

        /* Compute (L**H \ B) -> B   [ L**H \ (D \ (L \P**T * B)) ] */
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                    CblasUnit, n, nrhs, &ONE, A, lda, B, ldb);

        /* P * B  [ P * (L**H \ (D \ (L \P**T * B))) ] */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block */
                /* Interchange rows k and ipiv[k] (0-based). */
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 1;
            } else {
                /* 2 x 2 diagonal block */
                /* Interchange rows k and -ipiv[k]-1 (0-based). */
                kp = -ipiv[k] - 1;
                if (k > 0 && kp == -ipiv[k - 1] - 1) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 2;
            }
        }

    }

    csyconv(uplo, "R", n, A, lda, ipiv, work, &iinfo);
}
