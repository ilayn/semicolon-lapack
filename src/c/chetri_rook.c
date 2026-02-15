/**
 * @file chetri_rook.c
 * @brief CHETRI_ROOK computes the inverse of a complex Hermitian indefinite
 *        matrix using the factorization computed by CHETRF_ROOK.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRI_ROOK computes the inverse of a complex Hermitian indefinite matrix
 * A using the factorization A = U*D*U**H or A = L*D*L**H computed by
 * CHETRF_ROOK.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**H
 *                        = 'L': Lower triangular, A = L*D*L**H
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the factored matrix from CHETRF_ROOK.
 *                      On exit, the (Hermitian) inverse of the original matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from CHETRF_ROOK.
 * @param[out]    work  Complex*16 array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void chetri_rook(
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
        xerbla("CHETRI_ROOK", -(*info));
        return;
    }

    if (n == 0) return;

    /* Check that the diagonal matrix D is nonsingular. */
    if (upper) {
        for (int k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    } else {
        for (int k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    }

    if (upper) {

        int k = 0;
        while (k < n) {
            int kstep;

            if (ipiv[k] >= 0) {

                A[k + k * lda] = CMPLXF(ONE / crealf(A[k + k * lda]), 0.0f);

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

                f32 t = cabsf(A[k + (k + 1) * lda]);
                f32 ak = crealf(A[k + k * lda]) / t;
                f32 akp1 = crealf(A[(k + 1) + (k + 1) * lda]) / t;
                c64 akkp1 = A[k + (k + 1) * lda] / t;
                f32 d = t * (ak * akp1 - ONE);
                A[k + k * lda] = CMPLXF(akp1 / d, 0.0f);
                A[(k + 1) + (k + 1) * lda] = CMPLXF(ak / d, 0.0f);
                A[k + (k + 1) * lda] = -akkp1 / d;

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

            if (kstep == 1) {

                int kp = ipiv[k];
                if (kp != k) {

                    if (kp > 0)
                        cblas_cswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);

                    for (int j = kp + 1; j < k; j++) {
                        c64 temp = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conjf(A[kp + k * lda]);

                    c64 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                /* (1) Interchange rows and columns K and -IPIV(K) */
                int kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp > 0)
                        cblas_cswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);

                    for (int j = kp + 1; j < k; j++) {
                        c64 temp = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conjf(A[kp + k * lda]);

                    c64 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;

                    temp = A[k + (k + 1) * lda];
                    A[k + (k + 1) * lda] = A[kp + (k + 1) * lda];
                    A[kp + (k + 1) * lda] = temp;
                }

                /* (2) Interchange rows and columns K+1 and -IPIV(K+1) */
                k = k + 1;
                kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp > 0)
                        cblas_cswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);

                    for (int j = kp + 1; j < k; j++) {
                        c64 temp = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conjf(A[kp + k * lda]);

                    c64 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k + 1;
        }

    } else {

        int k = n - 1;
        while (k >= 0) {
            int kstep;

            if (ipiv[k] >= 0) {

                A[k + k * lda] = CMPLXF(ONE / crealf(A[k + k * lda]), 0.0f);

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

                f32 t = cabsf(A[k + (k - 1) * lda]);
                f32 ak = crealf(A[(k - 1) + (k - 1) * lda]) / t;
                f32 akp1 = crealf(A[k + k * lda]) / t;
                c64 akkp1 = A[k + (k - 1) * lda] / t;
                f32 d = t * (ak * akp1 - ONE);
                A[(k - 1) + (k - 1) * lda] = CMPLXF(akp1 / d, 0.0f);
                A[k + k * lda] = CMPLXF(ak / d, 0.0f);
                A[k + (k - 1) * lda] = -akkp1 / d;

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

            if (kstep == 1) {

                int kp = ipiv[k];
                if (kp != k) {

                    if (kp < n - 1)
                        cblas_cswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);

                    for (int j = k + 1; j < kp; j++) {
                        c64 temp = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conjf(A[kp + k * lda]);

                    c64 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                /* (1) Interchange rows and columns K and -IPIV(K) */
                int kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp < n - 1)
                        cblas_cswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);

                    for (int j = k + 1; j < kp; j++) {
                        c64 temp = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conjf(A[kp + k * lda]);

                    c64 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;

                    temp = A[k + (k - 1) * lda];
                    A[k + (k - 1) * lda] = A[kp + (k - 1) * lda];
                    A[kp + (k - 1) * lda] = temp;
                }

                /* (2) Interchange rows and columns K-1 and -IPIV(K-1) */
                k = k - 1;
                kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp < n - 1)
                        cblas_cswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);

                    for (int j = k + 1; j < kp; j++) {
                        c64 temp = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conjf(A[kp + k * lda]);

                    c64 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k - 1;
        }
    }
}
