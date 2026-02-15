/**
 * @file chetri.c
 * @brief CHETRI computes the inverse of a complex Hermitian indefinite matrix
 *        using the factorization computed by CHETRF.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRI computes the inverse of a complex Hermitian indefinite matrix
 * A using the factorization A = U*D*U**H or A = L*D*L**H computed by
 * CHETRF.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**H
 *                        = 'L': Lower triangular, A = L*D*L**H
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the factored matrix from CHETRF.
 *                      On exit, the (Hermitian) inverse of the original matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from CHETRF.
 * @param[out]    work  Complex*16 array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void chetri(
    const char* uplo,
    const int n,
    c64* restrict A,
    const int lda,
    const int* restrict ipiv,
    c64* restrict work,
    int* info)
{
    const f32 ONE = 1.0f;
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CHETRI", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) return;

    /* Check that the diagonal matrix D is nonsingular. */
    if (upper) {
        /* Upper triangular storage: examine D from bottom to top. */
        for (int k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    } else {
        /* Lower triangular storage: examine D from top to bottom. */
        for (int k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    }

    if (upper) {
        /* Compute inv(A) from the factorization A = U*D*U**H.
         *
         * K increases from 0 to n-1 in steps of 1 or 2. */
        int k = 0;
        while (k < n) {
            int kstep;
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block: invert it. */
                A[k + k * lda] = CMPLXF(ONE / crealf(A[k + k * lda]), 0.0f);

                /* Compute column k of the inverse. */
                if (k > 0) {
                    cblas_ccopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_chemv(CblasColMajor, CblasUpper,
                                k, &NEG_CONE, A, lda, work, 1,
                                &ZERO, &A[0 + k * lda], 1);
                    c64 dotc;
                    cblas_cdotc_sub(k, work, 1, &A[0 + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLXF(crealf(dotc), 0.0f);
                }
                kstep = 1;
            } else {
                /* 2x2 diagonal block: invert the 2x2 block. */
                f32 t = cabsf(A[k + (k + 1) * lda]);
                f32 ak = crealf(A[k + k * lda]) / t;
                f32 akp1 = crealf(A[(k + 1) + (k + 1) * lda]) / t;
                c64 akkp1 = A[k + (k + 1) * lda] / t;
                f32 d = t * (ak * akp1 - ONE);
                A[k + k * lda] = CMPLXF(akp1 / d, 0.0f);
                A[(k + 1) + (k + 1) * lda] = CMPLXF(ak / d, 0.0f);
                A[k + (k + 1) * lda] = -akkp1 / d;

                /* Compute columns k and k+1 of the inverse. */
                if (k > 0) {
                    cblas_ccopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_chemv(CblasColMajor, CblasUpper,
                                k, &NEG_CONE, A, lda, work, 1,
                                &ZERO, &A[0 + k * lda], 1);
                    c64 dotc;
                    cblas_cdotc_sub(k, work, 1, &A[0 + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLXF(crealf(dotc), 0.0f);
                    cblas_cdotc_sub(k, &A[0 + k * lda], 1, &A[0 + (k + 1) * lda], 1, &dotc);
                    A[k + (k + 1) * lda] = A[k + (k + 1) * lda] - dotc;
                    cblas_ccopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    cblas_chemv(CblasColMajor, CblasUpper,
                                k, &NEG_CONE, A, lda, work, 1,
                                &ZERO, &A[0 + (k + 1) * lda], 1);
                    cblas_cdotc_sub(k, work, 1, &A[0 + (k + 1) * lda], 1, &dotc);
                    A[(k + 1) + (k + 1) * lda] = A[(k + 1) + (k + 1) * lda] -
                                                   CMPLXF(crealf(dotc), 0.0f);
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp. */
            int kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                /* Swap columns k and kp in rows 0:kp-1 */
                if (kp > 0) {
                    cblas_cswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                }

                /* Swap and conjugate elements between kp and k */
                for (int j = kp + 1; j < k; j++) {
                    c64 temp = conjf(A[j + k * lda]);
                    A[j + k * lda] = conjf(A[kp + j * lda]);
                    A[kp + j * lda] = temp;
                }
                A[kp + k * lda] = conjf(A[kp + k * lda]);

                /* Swap diagonal elements */
                c64 temp = A[k + k * lda];
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
        /* Compute inv(A) from the factorization A = L*D*L**H.
         *
         * K decreases from n-1 to 0 in steps of 1 or 2. */
        int k = n - 1;
        while (k >= 0) {
            int kstep;
            if (ipiv[k] >= 0) {
                /* 1x1 diagonal block: invert it. */
                A[k + k * lda] = CMPLXF(ONE / crealf(A[k + k * lda]), 0.0f);

                /* Compute column k of the inverse. */
                if (k < n - 1) {
                    cblas_ccopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_chemv(CblasColMajor, CblasLower,
                                n - k - 1, &NEG_CONE, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, &ZERO, &A[(k + 1) + k * lda], 1);
                    c64 dotc;
                    cblas_cdotc_sub(n - k - 1, work, 1, &A[(k + 1) + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLXF(crealf(dotc), 0.0f);
                }
                kstep = 1;
            } else {
                /* 2x2 diagonal block: invert the 2x2 block. */
                f32 t = cabsf(A[k + (k - 1) * lda]);
                f32 ak = crealf(A[(k - 1) + (k - 1) * lda]) / t;
                f32 akp1 = crealf(A[k + k * lda]) / t;
                c64 akkp1 = A[k + (k - 1) * lda] / t;
                f32 d = t * (ak * akp1 - ONE);
                A[(k - 1) + (k - 1) * lda] = CMPLXF(akp1 / d, 0.0f);
                A[k + k * lda] = CMPLXF(ak / d, 0.0f);
                A[k + (k - 1) * lda] = -akkp1 / d;

                /* Compute columns k-1 and k of the inverse. */
                if (k < n - 1) {
                    cblas_ccopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_chemv(CblasColMajor, CblasLower,
                                n - k - 1, &NEG_CONE, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, &ZERO, &A[(k + 1) + k * lda], 1);
                    c64 dotc;
                    cblas_cdotc_sub(n - k - 1, work, 1, &A[(k + 1) + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLXF(crealf(dotc), 0.0f);
                    cblas_cdotc_sub(n - k - 1, &A[(k + 1) + k * lda], 1,
                                    &A[(k + 1) + (k - 1) * lda], 1, &dotc);
                    A[k + (k - 1) * lda] = A[k + (k - 1) * lda] - dotc;
                    cblas_ccopy(n - k - 1, &A[(k + 1) + (k - 1) * lda], 1, work, 1);
                    cblas_chemv(CblasColMajor, CblasLower,
                                n - k - 1, &NEG_CONE, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, &ZERO, &A[(k + 1) + (k - 1) * lda], 1);
                    cblas_cdotc_sub(n - k - 1, work, 1,
                                    &A[(k + 1) + (k - 1) * lda], 1, &dotc);
                    A[(k - 1) + (k - 1) * lda] = A[(k - 1) + (k - 1) * lda] -
                                                   CMPLXF(crealf(dotc), 0.0f);
                }
                kstep = 2;
            }

            /* Interchange rows and columns k and kp. */
            int kp;
            if (ipiv[k] >= 0) {
                kp = ipiv[k];
            } else {
                kp = -(ipiv[k] + 1);
            }

            if (kp != k) {
                /* Swap columns k and kp in rows kp+1:n-1 */
                if (kp < n - 1) {
                    cblas_cswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                &A[(kp + 1) + kp * lda], 1);
                }

                /* Swap and conjugate elements between k and kp */
                for (int j = k + 1; j < kp; j++) {
                    c64 temp = conjf(A[j + k * lda]);
                    A[j + k * lda] = conjf(A[kp + j * lda]);
                    A[kp + j * lda] = temp;
                }
                A[kp + k * lda] = conjf(A[kp + k * lda]);

                /* Swap diagonal elements */
                c64 temp = A[k + k * lda];
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
