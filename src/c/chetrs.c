/**
 * @file chetrs.c
 * @brief CHETRS solves a system of linear equations A*X = B with a complex
 *        Hermitian matrix A using the factorization computed by CHETRF.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRS solves a system of linear equations A*X = B with a complex
 * Hermitian matrix A using the factorization A = U*D*U**H or
 * A = L*D*L**H computed by CHETRF.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**H
 *                        = 'L': Lower triangular, A = L*D*L**H
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     A     Complex*16 array, dimension (lda, n).
 *                      The block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L as computed by CHETRF.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from CHETRF.
 * @param[in,out] B     Complex*16 array, dimension (ldb, nrhs).
 *                      On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void chetrs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c64* restrict B,
    const INT ldb,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);

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
        xerbla("CHETRS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) return;

    if (upper) {

        /* Solve A*X = B, where A = U*D*U**H.
         *
         * First solve U*D*X = B, overwriting B with X.
         *
         * K is the main loop index, decreasing from n-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        INT k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(U(k)), where U(k) is stored in column k. */
                if (k > 0) {
                    cblas_cgeru(CblasColMajor, k, nrhs,
                                &NEG_ONE, &A[0 + k * lda], 1, &B[k], ldb,
                                &B[0], ldb);
                }

                /* Multiply by inverse of diagonal block. */
                f32 s = 1.0f / crealf(A[k + k * lda]);
                cblas_csscal(nrhs, s, &B[k], ldb);
                k--;
            } else {
                /* 2x2 diagonal block.
                 * Interchange rows k-1 and -(ipiv[k]+1). */
                INT kp = -(ipiv[k] + 1);
                if (kp != k - 1) {
                    cblas_cswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(U(k)), stored in columns k-1 and k. */
                if (k > 1) {
                    cblas_cgeru(CblasColMajor, k - 1, nrhs,
                                &NEG_ONE, &A[0 + k * lda], 1, &B[k], ldb,
                                &B[0], ldb);
                    cblas_cgeru(CblasColMajor, k - 1, nrhs,
                                &NEG_ONE, &A[0 + (k - 1) * lda], 1, &B[k - 1], ldb,
                                &B[0], ldb);
                }

                /* Multiply by inverse of 2x2 diagonal block. */
                c64 akm1k = A[(k - 1) + k * lda];
                c64 akm1 = A[(k - 1) + (k - 1) * lda] / akm1k;
                c64 ak = A[k + k * lda] / conjf(akm1k);
                c64 denom = akm1 * ak - ONE;
                for (INT j = 0; j < nrhs; j++) {
                    c64 bkm1 = B[(k - 1) + j * ldb] / akm1k;
                    c64 bk = B[k + j * ldb] / conjf(akm1k);
                    B[(k - 1) + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[k + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k -= 2;
            }
        }

        /* Next solve U**H * X = B, overwriting B with X.
         *
         * K increases from 0 to n-1 in steps of 1 or 2. */
        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Multiply by inv(U**H(k)). */
                if (k > 0) {
                    clacgv(nrhs, &B[k], ldb);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                k, nrhs,
                                &NEG_ONE, &B[0], ldb, &A[0 + k * lda], 1,
                                &ONE, &B[k], ldb);
                    clacgv(nrhs, &B[k], ldb);
                }

                /* Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k++;
            } else {
                /* 2x2 diagonal block.
                 * Multiply by inv(U**H(k+1)). */
                if (k > 0) {
                    clacgv(nrhs, &B[k], ldb);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                k, nrhs,
                                &NEG_ONE, &B[0], ldb, &A[0 + k * lda], 1,
                                &ONE, &B[k], ldb);
                    clacgv(nrhs, &B[k], ldb);

                    clacgv(nrhs, &B[k + 1], ldb);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                k, nrhs,
                                &NEG_ONE, &B[0], ldb, &A[0 + (k + 1) * lda], 1,
                                &ONE, &B[k + 1], ldb);
                    clacgv(nrhs, &B[k + 1], ldb);
                }

                /* Interchange rows k and -(ipiv[k]+1). */
                INT kp = -(ipiv[k] + 1);
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k += 2;
            }
        }

    } else {

        /* Solve A*X = B, where A = L*D*L**H.
         *
         * First solve L*D*X = B, overwriting B with X.
         *
         * K increases from 0 to n-1 in steps of 1 or 2. */
        INT k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(L(k)). */
                if (k < n - 1) {
                    cblas_cgeru(CblasColMajor, n - k - 1, nrhs,
                                &NEG_ONE, &A[(k + 1) + k * lda], 1, &B[k], ldb,
                                &B[k + 1], ldb);
                }

                /* Multiply by inverse of diagonal block. */
                f32 s = 1.0f / crealf(A[k + k * lda]);
                cblas_csscal(nrhs, s, &B[k], ldb);
                k++;
            } else {
                /* 2x2 diagonal block.
                 * Interchange rows k+1 and -(ipiv[k]+1). */
                INT kp = -(ipiv[k] + 1);
                if (kp != k + 1) {
                    cblas_cswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }

                /* Multiply by inv(L(k)). */
                if (k < n - 2) {
                    cblas_cgeru(CblasColMajor, n - k - 2, nrhs,
                                &NEG_ONE, &A[(k + 2) + k * lda], 1, &B[k], ldb,
                                &B[k + 2], ldb);
                    cblas_cgeru(CblasColMajor, n - k - 2, nrhs,
                                &NEG_ONE, &A[(k + 2) + (k + 1) * lda], 1, &B[k + 1], ldb,
                                &B[k + 2], ldb);
                }

                /* Multiply by inverse of 2x2 diagonal block. */
                c64 akm1k = A[(k + 1) + k * lda];
                c64 akm1 = A[k + k * lda] / conjf(akm1k);
                c64 ak = A[(k + 1) + (k + 1) * lda] / akm1k;
                c64 denom = akm1 * ak - ONE;
                for (INT j = 0; j < nrhs; j++) {
                    c64 bkm1 = B[k + j * ldb] / conjf(akm1k);
                    c64 bk = B[(k + 1) + j * ldb] / akm1k;
                    B[k + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[(k + 1) + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k += 2;
            }
        }

        /* Next solve L**H * X = B, overwriting B with X.
         *
         * K decreases from n-1 to 0 in steps of 1 or 2. */
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block.
                 * Multiply by inv(L**H(k)). */
                if (k < n - 1) {
                    clacgv(nrhs, &B[k], ldb);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                n - k - 1, nrhs,
                                &NEG_ONE, &B[k + 1], ldb, &A[(k + 1) + k * lda], 1,
                                &ONE, &B[k], ldb);
                    clacgv(nrhs, &B[k], ldb);
                }

                /* Interchange rows k and ipiv[k]. */
                INT kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k--;
            } else {
                /* 2x2 diagonal block.
                 * Multiply by inv(L**H(k-1)). */
                if (k < n - 1) {
                    clacgv(nrhs, &B[k], ldb);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                n - k - 1, nrhs,
                                &NEG_ONE, &B[k + 1], ldb, &A[(k + 1) + k * lda], 1,
                                &ONE, &B[k], ldb);
                    clacgv(nrhs, &B[k], ldb);

                    clacgv(nrhs, &B[k - 1], ldb);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                n - k - 1, nrhs,
                                &NEG_ONE, &B[k + 1], ldb, &A[(k + 1) + (k - 1) * lda], 1,
                                &ONE, &B[k - 1], ldb);
                    clacgv(nrhs, &B[k - 1], ldb);
                }

                /* Interchange rows k and -(ipiv[k]+1). */
                INT kp = -(ipiv[k] + 1);
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k -= 2;
            }
        }
    }
}
