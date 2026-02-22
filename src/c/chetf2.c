/**
 * @file chetf2.c
 * @brief CHETF2 computes the factorization of a complex Hermitian matrix,
 *        using the diagonal pivoting method (unblocked algorithm, calling
 *        Level 2 BLAS).
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/* Alpha for Bunch-Kaufman pivoting: (1 + sqrt(17)) / 8 */
static const f32 ALPHA_BK = 0.6403882032022076f;

/**
 * CHETF2 computes the factorization of a complex Hermitian matrix A
 * using the Bunch-Kaufman diagonal pivoting method:
 *
 *    A = U*D*U**H  or  A = L*D*L**H
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, U**H is the conjugate transpose of U, and D is
 * Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo  'U': Upper triangular factorization (A = U*D*U**H)
 *                      'L': Lower triangular factorization (A = L*D*L**H)
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A. If uplo = "U", the
 *                      leading n-by-n upper triangular part contains the upper
 *                      triangular part of A. If uplo = "L", the leading n-by-n
 *                      lower triangular part contains the lower triangular part.
 *                      On exit, the block diagonal matrix D and the multipliers
 *                      used to obtain the factor U or L.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[out]    ipiv  Integer array, dimension (n). Pivot indices (0-based).
 *                      If ipiv[k] >= 0: rows/columns k and ipiv[k] were
 *                          interchanged, D(k,k) is a 1-by-1 block.
 *                      If ipiv[k] < 0 (upper): rows/columns k-1 and
 *                          -ipiv[k]-1 were interchanged,
 *                          D(k-1:k,k-1:k) is a 2-by-2 block,
 *                          and ipiv[k-1] = ipiv[k].
 *                      If ipiv[k] < 0 (lower): rows/columns k+1 and
 *                          -ipiv[k]-1 were interchanged,
 *                          D(k:k+1,k:k+1) is a 2-by-2 block,
 *                          and ipiv[k+1] = ipiv[k].
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = k, D(k,k) is exactly zero. The
 *                           factorization has been completed, but D is exactly
 *                           singular, and division by zero will occur if it is
 *                           used to solve a system of equations.
 */
void chetf2(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    /* Test the input parameters */
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    *info = 0;
    if (!upper && uplo[0] != 'L' && uplo[0] != 'l') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CHETF2", -*info);
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    if (upper) {
        /*
         * Factorize A as U*D*U**H using the upper triangle of A
         *
         * K is the main loop index, decreasing from N to 1 in steps of
         * 1 or 2
         */

        INT k = n - 1;
        while (k >= 0) {
            INT kstep = 1;
            f32 absakk = fabsf(crealf(A[k + k * lda]));

            INT imax = 0;
            f32 colmax = 0.0f;
            if (k > 0) {
                imax = cblas_icamax(k, &A[0 + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            }

            if (fmaxf(absakk, colmax) == 0.0f || isnan(absakk)) {
                if (*info == 0) {
                    *info = k + 1;
                }
                ipiv[k] = k;
                A[k + k * lda] = crealf(A[k + k * lda]);
            } else {
                INT kp;
                if (absakk >= ALPHA_BK * colmax) {
                    kp = k;
                } else {
                    f32 rowmax = 0.0f;
                    if (imax + 1 <= k) {
                        INT jmax = imax + 1 + cblas_icamax(k - imax, &A[imax + (imax + 1) * lda], lda);
                        rowmax = cabs1f(A[imax + jmax * lda]);
                    }
                    if (imax > 0) {
                        INT jmax = cblas_icamax(imax, &A[0 + imax * lda], 1);
                        rowmax = fmaxf(rowmax, cabs1f(A[jmax + imax * lda]));
                    }

                    if (absakk >= ALPHA_BK * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (fabsf(crealf(A[imax + imax * lda])) >= ALPHA_BK * rowmax) {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }

                INT kk = k - kstep + 1;
                if (kp != kk) {
                    if (kp > 0) {
                        cblas_cswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    for (INT j = kp + 1; j < kk; j++) {
                        c64 t = conjf(A[j + kk * lda]);
                        A[j + kk * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = t;
                    }
                    A[kp + kk * lda] = conjf(A[kp + kk * lda]);
                    f32 r1 = crealf(A[kk + kk * lda]);
                    A[kk + kk * lda] = crealf(A[kp + kp * lda]);
                    A[kp + kp * lda] = r1;
                    if (kstep == 2) {
                        A[k + k * lda] = crealf(A[k + k * lda]);
                        c64 t = A[(k - 1) + k * lda];
                        A[(k - 1) + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                } else {
                    A[k + k * lda] = crealf(A[k + k * lda]);
                    if (kstep == 2)
                        A[(k - 1) + (k - 1) * lda] = crealf(A[(k - 1) + (k - 1) * lda]);
                }

                if (kstep == 1) {
                    if (k > 0) {
                        f32 r1 = 1.0f / crealf(A[k + k * lda]);
                        cblas_cher(CblasColMajor, CblasUpper, k, -r1,
                                   &A[0 + k * lda], 1, A, lda);
                        cblas_csscal(k, r1, &A[0 + k * lda], 1);
                    }
                } else {
                    if (k > 1) {
                        f32 d = slapy2(crealf(A[(k - 1) + k * lda]),
                                          cimagf(A[(k - 1) + k * lda]));
                        f32 d22 = crealf(A[(k - 1) + (k - 1) * lda]) / d;
                        f32 d11 = crealf(A[k + k * lda]) / d;
                        f32 tt = 1.0f / (d11 * d22 - 1.0f);
                        c64 d12 = A[(k - 1) + k * lda] / d;
                        d = tt / d;

                        for (INT j = k - 2; j >= 0; j--) {
                            c64 wkm1 = d * (d11 * A[j + (k - 1) * lda] -
                                                       conjf(d12) * A[j + k * lda]);
                            c64 wk = d * (d22 * A[j + k * lda] -
                                                     d12 * A[j + (k - 1) * lda]);
                            for (INT i = j; i >= 0; i--) {
                                A[i + j * lda] -= A[i + k * lda] * conjf(wk) +
                                                  A[i + (k - 1) * lda] * conjf(wkm1);
                            }
                            A[j + k * lda] = wk;
                            A[j + (k - 1) * lda] = wkm1;
                            A[j + j * lda] = CMPLXF(crealf(A[j + j * lda]), 0.0f);
                        }
                    }
                }

                if (kstep == 1) {
                    ipiv[k] = kp;
                } else {
                    ipiv[k] = -(kp + 1);
                    ipiv[k - 1] = -(kp + 1);
                }
            }

            k -= kstep;
        }

    } else {
        /*
         * Factorize A as L*D*L**H using the lower triangle of A
         *
         * K is the main loop index, increasing from 1 to N in steps of
         * 1 or 2
         */

        INT k = 0;
        while (k < n) {
            INT kstep = 1;
            f32 absakk = fabsf(crealf(A[k + k * lda]));

            INT imax = k;
            f32 colmax = 0.0f;
            if (k < n - 1) {
                imax = k + 1 + cblas_icamax(n - k - 1, &A[(k + 1) + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            }

            if (fmaxf(absakk, colmax) == 0.0f || isnan(absakk)) {
                if (*info == 0) {
                    *info = k + 1;
                }
                ipiv[k] = k;
                A[k + k * lda] = crealf(A[k + k * lda]);
            } else {
                INT kp;
                if (absakk >= ALPHA_BK * colmax) {
                    kp = k;
                } else {
                    f32 rowmax = 0.0f;
                    if (imax > k) {
                        INT jmax = k + cblas_icamax(imax - k, &A[imax + k * lda], lda);
                        rowmax = cabs1f(A[imax + jmax * lda]);
                    }
                    if (imax < n - 1) {
                        INT jmax = imax + 1 + cblas_icamax(n - imax - 1, &A[(imax + 1) + imax * lda], 1);
                        rowmax = fmaxf(rowmax, cabs1f(A[jmax + imax * lda]));
                    }

                    if (absakk >= ALPHA_BK * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (fabsf(crealf(A[imax + imax * lda])) >= ALPHA_BK * rowmax) {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }

                INT kk = k + kstep - 1;
                if (kp != kk) {
                    if (kp < n - 1) {
                        cblas_cswap(n - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);
                    }
                    for (INT j = kk + 1; j < kp; j++) {
                        c64 t = conjf(A[j + kk * lda]);
                        A[j + kk * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = t;
                    }
                    A[kp + kk * lda] = conjf(A[kp + kk * lda]);
                    f32 r1 = crealf(A[kk + kk * lda]);
                    A[kk + kk * lda] = crealf(A[kp + kp * lda]);
                    A[kp + kp * lda] = r1;
                    if (kstep == 2) {
                        A[k + k * lda] = crealf(A[k + k * lda]);
                        c64 t = A[(k + 1) + k * lda];
                        A[(k + 1) + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                } else {
                    A[k + k * lda] = crealf(A[k + k * lda]);
                    if (kstep == 2)
                        A[(k + 1) + (k + 1) * lda] = crealf(A[(k + 1) + (k + 1) * lda]);
                }

                if (kstep == 1) {
                    if (k < n - 1) {
                        f32 r1 = 1.0f / crealf(A[k + k * lda]);
                        cblas_cher(CblasColMajor, CblasLower, n - k - 1, -r1,
                                   &A[(k + 1) + k * lda], 1,
                                   &A[(k + 1) + (k + 1) * lda], lda);
                        cblas_csscal(n - k - 1, r1, &A[(k + 1) + k * lda], 1);
                    }
                } else {
                    if (k < n - 2) {
                        f32 d = slapy2(crealf(A[(k + 1) + k * lda]),
                                          cimagf(A[(k + 1) + k * lda]));
                        f32 d11 = crealf(A[(k + 1) + (k + 1) * lda]) / d;
                        f32 d22 = crealf(A[k + k * lda]) / d;
                        f32 tt = 1.0f / (d11 * d22 - 1.0f);
                        c64 d21 = A[(k + 1) + k * lda] / d;
                        d = tt / d;

                        for (INT j = k + 2; j < n; j++) {
                            c64 wk = d * (d11 * A[j + k * lda] -
                                                     d21 * A[j + (k + 1) * lda]);
                            c64 wkp1 = d * (d22 * A[j + (k + 1) * lda] -
                                                       conjf(d21) * A[j + k * lda]);
                            for (INT i = j; i < n; i++) {
                                A[i + j * lda] -= A[i + k * lda] * conjf(wk) +
                                                  A[i + (k + 1) * lda] * conjf(wkp1);
                            }
                            A[j + k * lda] = wk;
                            A[j + (k + 1) * lda] = wkp1;
                            A[j + j * lda] = CMPLXF(crealf(A[j + j * lda]), 0.0f);
                        }
                    }
                }

                if (kstep == 1) {
                    ipiv[k] = kp;
                } else {
                    ipiv[k] = -(kp + 1);
                    ipiv[k + 1] = -(kp + 1);
                }
            }

            k += kstep;
        }
    }
}
