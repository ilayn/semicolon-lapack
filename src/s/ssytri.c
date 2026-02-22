/**
 * @file ssytri.c
 * @brief SSYTRI computes the inverse of a real symmetric indefinite matrix
 *        using the factorization computed by SSYTRF.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SSYTRI computes the inverse of a real symmetric indefinite matrix
 * A using the factorization A = U*D*U**T or A = L*D*L**T computed by
 * SSYTRF.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**T
 *                        = 'L': Lower triangular, A = L*D*L**T
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the factored matrix from SSYTRF.
 *                      On exit, the symmetric inverse of the original matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from SSYTRF.
 * @param[out]    work  Double precision array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void ssytri(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    f32* restrict work,
    INT* info)
{
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SSYTRI", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) return;

    /* Check that the diagonal matrix D is nonsingular. */
    if (upper) {
        /* Upper triangular storage: examine D from bottom to top.
         * Fortran: DO INFO = N, 1, -1: checks IPIV(INFO)>0 .AND. A(INFO,INFO)==0
         * 0-based: check ipiv[k] >= 0 && A[k,k] == 0 */
        for (INT k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && A[k + k * lda] == 0.0f) {
                *info = k + 1;  /* 1-based info */
                return;
            }
        }
    } else {
        /* Lower triangular storage: examine D from top to bottom. */
        for (INT k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && A[k + k * lda] == 0.0f) {
                *info = k + 1;
                return;
            }
        }
    }

    if (upper) {
        /* Compute inv(A) from the factorization A = U*D*U**T.
         *
         * K increases from 0 to n-1 in steps of 1 or 2. */
        INT k = 0;
        while (k < n) {
            INT kstep;
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block: invert it. */
                A[k + k * lda] = 1.0f / A[k + k * lda];

                /* Compute column k of the inverse.
                 * Fortran: DCOPY(K-1, A(1,K), 1, WORK, 1)
                 *          DSYMV(UPLO, K-1, -1, A, LDA, WORK, 1, 0, A(1,K), 1)
                 *          A(K,K) -= DDOT(K-1, WORK, 1, A(1,K), 1)
                 * 0-based: count = k */
                if (k > 0) {
                    cblas_scopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasUpper,
                                k, -1.0f, A, lda, work, 1,
                                0.0f, &A[0 + k * lda], 1);
                    A[k + k * lda] -= cblas_sdot(k, work, 1, &A[0 + k * lda], 1);
                }
                kstep = 1;
            } else {
                /* 2x2 diagonal block: invert the 2x2 block.
                 * Fortran: T = ABS(A(K, K+1))
                 * 0-based: t = |A[k + (k+1)*lda]| */
                f32 t = fabsf(A[k + (k + 1) * lda]);
                f32 ak = A[k + k * lda] / t;
                f32 akp1 = A[(k + 1) + (k + 1) * lda] / t;
                f32 akkp1 = A[k + (k + 1) * lda] / t;
                f32 d = t * (ak * akp1 - 1.0f);
                A[k + k * lda] = akp1 / d;
                A[(k + 1) + (k + 1) * lda] = ak / d;
                A[k + (k + 1) * lda] = -akkp1 / d;

                /* Compute columns k and k+1 of the inverse.
                 * 0-based: count = k */
                if (k > 0) {
                    cblas_scopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasUpper,
                                k, -1.0f, A, lda, work, 1,
                                0.0f, &A[0 + k * lda], 1);
                    A[k + k * lda] -= cblas_sdot(k, work, 1, &A[0 + k * lda], 1);
                    A[k + (k + 1) * lda] -= cblas_sdot(k, &A[0 + k * lda], 1,
                                                        &A[0 + (k + 1) * lda], 1);
                    cblas_scopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasUpper,
                                k, -1.0f, A, lda, work, 1,
                                0.0f, &A[0 + (k + 1) * lda], 1);
                    A[(k + 1) + (k + 1) * lda] -= cblas_sdot(k, work, 1,
                                                              &A[0 + (k + 1) * lda], 1);
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp.
             * Fortran: KP = ABS(IPIV(K)) (1-based)
             * 0-based: for 1x1, kp = ipiv[k]; for 2x2, kp = -(ipiv[k]+1) */
            INT kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                /* Fortran: DSWAP(KP-1, A(1,K), 1, A(1,KP), 1) — count KP-1 (1-based) = kp
                 * 0-based: swap A(0:kp-1, k) with A(0:kp-1, kp), count = kp */
                if (kp > 0) {
                    cblas_sswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                }

                /* Fortran: DSWAP(K-KP-1, A(KP+1,K), 1, A(KP,KP+1), LDA)
                 * 0-based: swap A(kp+1:k-1, k) with A(kp, kp+1:k-1), count = k-kp-1 */
                if (k - kp - 1 > 0) {
                    cblas_sswap(k - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[kp + (kp + 1) * lda], lda);
                }

                /* Swap diagonal elements */
                f32 temp = A[k + k * lda];
                A[k + k * lda] = A[kp + kp * lda];
                A[kp + kp * lda] = temp;

                /* For 2x2 block, also swap A(k, k+1) with A(kp, k+1) */
                if (kstep == 2) {
                    temp = A[k + (k + 1) * lda];
                    A[k + (k + 1) * lda] = A[kp + (k + 1) * lda];
                    A[kp + (k + 1) * lda] = temp;
                }
            }

            k += kstep;
        }

    } else {
        /* Compute inv(A) from the factorization A = L*D*L**T.
         *
         * K decreases from n-1 to 0 in steps of 1 or 2. */
        INT k = n - 1;
        while (k >= 0) {
            INT kstep;
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block: invert it. */
                A[k + k * lda] = 1.0f / A[k + k * lda];

                /* Compute column k of the inverse.
                 * Fortran: DCOPY(N-K, A(K+1,K), 1, WORK, 1)
                 *          DSYMV(UPLO, N-K, -1, A(K+1,K+1), LDA, WORK, 1, 0, A(K+1,K), 1)
                 *          A(K,K) -= DDOT(N-K, WORK, 1, A(K+1,K), 1)
                 * 0-based: count = n-k-1, submatrix starts at (k+1, k+1) */
                if (k < n - 1) {
                    cblas_scopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasLower,
                                n - k - 1, -1.0f, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, 0.0f, &A[(k + 1) + k * lda], 1);
                    A[k + k * lda] -= cblas_sdot(n - k - 1, work, 1,
                                                  &A[(k + 1) + k * lda], 1);
                }
                kstep = 1;
            } else {
                /* 2x2 diagonal block: invert the 2x2 block.
                 * Fortran: T = ABS(A(K, K-1))
                 * 0-based: t = |A[k + (k-1)*lda]| */
                f32 t = fabsf(A[k + (k - 1) * lda]);
                f32 ak = A[(k - 1) + (k - 1) * lda] / t;
                f32 akp1 = A[k + k * lda] / t;
                f32 akkp1 = A[k + (k - 1) * lda] / t;
                f32 d = t * (ak * akp1 - 1.0f);
                A[(k - 1) + (k - 1) * lda] = akp1 / d;
                A[k + k * lda] = ak / d;
                A[k + (k - 1) * lda] = -akkp1 / d;

                /* Compute columns k-1 and k of the inverse.
                 * 0-based: count = n-k-1, submatrix at (k+1, k+1) */
                if (k < n - 1) {
                    cblas_scopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasLower,
                                n - k - 1, -1.0f, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, 0.0f, &A[(k + 1) + k * lda], 1);
                    A[k + k * lda] -= cblas_sdot(n - k - 1, work, 1,
                                                  &A[(k + 1) + k * lda], 1);
                    A[k + (k - 1) * lda] -= cblas_sdot(n - k - 1,
                                                        &A[(k + 1) + k * lda], 1,
                                                        &A[(k + 1) + (k - 1) * lda], 1);
                    cblas_scopy(n - k - 1, &A[(k + 1) + (k - 1) * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasLower,
                                n - k - 1, -1.0f, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, 0.0f, &A[(k + 1) + (k - 1) * lda], 1);
                    A[(k - 1) + (k - 1) * lda] -= cblas_sdot(n - k - 1, work, 1,
                                                              &A[(k + 1) + (k - 1) * lda], 1);
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp. */
            INT kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                /* Fortran: DSWAP(N-KP, A(KP+1,K), 1, A(KP+1,KP), 1)
                 * 0-based: count = n-kp-1 */
                if (kp < n - 1) {
                    cblas_sswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[(kp + 1) + kp * lda], 1);
                }

                /* Fortran: DSWAP(KP-K-1, A(K+1,K), 1, A(KP,K+1), LDA)
                 * 0-based: count = kp-k-1 */
                if (kp - k - 1 > 0) {
                    cblas_sswap(kp - k - 1, &A[(k + 1) + k * lda], 1,
                                &A[kp + (k + 1) * lda], lda);
                }

                /* Swap diagonal elements */
                f32 temp = A[k + k * lda];
                A[k + k * lda] = A[kp + kp * lda];
                A[kp + kp * lda] = temp;

                /* For 2x2 block, also swap A(k, k-1) with A(kp, k-1) */
                if (kstep == 2) {
                    temp = A[k + (k - 1) * lda];
                    A[k + (k - 1) * lda] = A[kp + (k - 1) * lda];
                    A[kp + (k - 1) * lda] = temp;
                }
            }

            k -= kstep;
        }
    }
}
