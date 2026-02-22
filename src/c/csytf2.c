/**
 * @file csytf2.c
 * @brief CSYTF2 computes the factorization of a complex symmetric indefinite
 *        matrix using the Bunch-Kaufman diagonal pivoting method (unblocked).
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/* Alpha for Bunch-Kaufman pivoting: (1 + sqrt(17)) / 8 */
static const f32 ALPHA_BK = 0.6403882032022076f;

/**
 * CSYTF2 computes the factorization of a complex symmetric matrix A using
 * the Bunch-Kaufman diagonal pivoting method:
 *
 *    A = U*D*U**T  or  A = L*D*L**T
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, U**T is the transpose of U, and D is symmetric and
 * block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo  'U': Upper triangular factorization (A = U*D*U**T)
 *                      'L': Lower triangular factorization (A = L*D*L**T)
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the symmetric matrix A. If uplo = "U", the
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
void csytf2(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);

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
        xerbla("CSYTF2", -*info);
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    if (upper) {
        /* Factorize A as U*D*U**T using the upper triangle of A
         * k decreases from n-1 to 0 in steps of 1 or 2 */

        INT k = n - 1;
        while (k >= 0) {
            INT kstep = 1;
            f32 absakk = cabs1f(A[k + k * lda]);

            /* Find largest off-diagonal element in column k */
            INT imax = 0;
            f32 colmax = 0.0f;
            if (k > 0) {
                imax = cblas_icamax(k, &A[0 + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            }

            if (fmaxf(absakk, colmax) == 0.0f || isnan(absakk)) {
                /* Column k is zero or underflow, or contains a NaN:
                 * set info and continue */
                if (*info == 0) {
                    *info = k + 1;
                }
                ipiv[k] = k;
            } else {
                INT kp;
                if (absakk >= ALPHA_BK * colmax) {
                    /* No interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* JMAX is the column-index of the largest off-diagonal
                     * element in row imax, and rowmax is its absolute value */
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
                        /* No interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (cabs1f(A[imax + imax * lda]) >= ALPHA_BK * rowmax) {
                        /* Interchange rows and columns k and imax,
                         * use 1-by-1 pivot block */
                        kp = imax;
                    } else {
                        /* Interchange rows and columns k-1 and imax,
                         * use 2-by-2 pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                INT kk = k - kstep + 1;
                if (kp != kk) {
                    /* Interchange rows and columns kk and kp
                     * in the leading submatrix A(0:k, 0:k) */
                    if (kp > 0) {
                        cblas_cswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    if (kk - kp - 1 > 0) {
                        cblas_cswap(kk - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[kp + (kp + 1) * lda], lda);
                    }
                    c64 t = A[kk + kk * lda];
                    A[kk + kk * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = t;
                    if (kstep == 2) {
                        t = A[(k - 1) + k * lda];
                        A[(k - 1) + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                }

                /* Update the leading submatrix */
                if (kstep == 1) {
                    /* 1-by-1 pivot block D(k): column k now holds
                     * W(k) = U(k)*D(k), where U(k) is the k-th column of U.
                     * Perform a rank-1 update of A(0:k-1, 0:k-1) as
                     * A := A - U(k)*D(k)*U(k)**T = A - W(k)*(1/D(k))*W(k)**T */
                    if (k > 0) {
                        c64 r1 = CONE / A[k + k * lda];
                        c64 neg_r1 = -r1;
                        csyr(uplo, k, neg_r1,
                             &A[0 + k * lda], 1, A, lda);
                        cblas_cscal(k, &r1, &A[0 + k * lda], 1);
                    }
                } else {
                    /* 2-by-2 pivot block D(k): columns k and k-1 now hold
                     * (W(k-1) W(k)) = (U(k-1) U(k))*D(k).
                     * Perform a rank-2 update of A(0:k-2, 0:k-2) as
                     * A := A - (W(k-1) W(k))*inv(D(k))*(W(k-1) W(k))**T */
                    if (k > 1) {
                        c64 d12 = A[(k - 1) + k * lda];
                        c64 d22 = A[(k - 1) + (k - 1) * lda] / d12;
                        c64 d11 = A[k + k * lda] / d12;
                        c64 t = CONE / (d11 * d22 - CONE);
                        d12 = t / d12;

                        for (INT j = k - 2; j >= 0; j--) {
                            c64 wkm1 = d12 * (d11 * A[j + (k - 1) * lda] - A[j + k * lda]);
                            c64 wk = d12 * (d22 * A[j + k * lda] - A[j + (k - 1) * lda]);
                            for (INT i = j; i >= 0; i--) {
                                A[i + j * lda] -= A[i + k * lda] * wk +
                                                  A[i + (k - 1) * lda] * wkm1;
                            }
                            A[j + k * lda] = wk;
                            A[j + (k - 1) * lda] = wkm1;
                        }
                    }
                }

                /* Store details of the interchanges in ipiv */
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
        /* Factorize A as L*D*L**T using the lower triangle of A
         * k increases from 0 to n-1 in steps of 1 or 2 */

        INT k = 0;
        while (k < n) {
            INT kstep = 1;
            f32 absakk = cabs1f(A[k + k * lda]);

            /* Find largest off-diagonal element in column k */
            INT imax = k;
            f32 colmax = 0.0f;
            if (k < n - 1) {
                imax = k + 1 + cblas_icamax(n - k - 1, &A[(k + 1) + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            }

            if (fmaxf(absakk, colmax) == 0.0f || isnan(absakk)) {
                /* Column k is zero or underflow, or contains a NaN:
                 * set info and continue */
                if (*info == 0) {
                    *info = k + 1;
                }
                ipiv[k] = k;
            } else {
                INT kp;
                if (absakk >= ALPHA_BK * colmax) {
                    /* No interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* JMAX is the column-index of the largest off-diagonal
                     * element in row imax, and rowmax is its absolute value */
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
                        /* No interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (cabs1f(A[imax + imax * lda]) >= ALPHA_BK * rowmax) {
                        /* Interchange rows and columns k and imax,
                         * use 1-by-1 pivot block */
                        kp = imax;
                    } else {
                        /* Interchange rows and columns k+1 and imax,
                         * use 2-by-2 pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                INT kk = k + kstep - 1;
                if (kp != kk) {
                    /* Interchange rows and columns kk and kp
                     * in the trailing submatrix A(k:n-1, k:n-1) */
                    if (kp < n - 1) {
                        cblas_cswap(n - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);
                    }
                    if (kp - kk - 1 > 0) {
                        cblas_cswap(kp - kk - 1, &A[(kk + 1) + kk * lda], 1,
                                    &A[kp + (kk + 1) * lda], lda);
                    }
                    c64 t = A[kk + kk * lda];
                    A[kk + kk * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = t;
                    if (kstep == 2) {
                        t = A[(k + 1) + k * lda];
                        A[(k + 1) + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                }

                /* Update the trailing submatrix */
                if (kstep == 1) {
                    /* 1-by-1 pivot block D(k): column k now holds
                     * W(k) = L(k)*D(k), where L(k) is the k-th column of L.
                     * Perform a rank-1 update of A(k+1:n-1, k+1:n-1) as
                     * A := A - L(k)*D(k)*L(k)**T = A - W(k)*(1/D(k))*W(k)**T */
                    if (k < n - 1) {
                        c64 d11 = CONE / A[k + k * lda];
                        c64 neg_d11 = -d11;
                        csyr(uplo, n - k - 1, neg_d11,
                             &A[(k + 1) + k * lda], 1, &A[(k + 1) + (k + 1) * lda], lda);
                        cblas_cscal(n - k - 1, &d11, &A[(k + 1) + k * lda], 1);
                    }
                } else {
                    /* 2-by-2 pivot block D(k): columns k and k+1 now hold
                     * (W(k) W(k+1)) = (L(k) L(k+1))*D(k).
                     * Perform a rank-2 update of A(k+2:n-1, k+2:n-1) as
                     * A := A - (W(k) W(k+1))*inv(D(k))*(W(k) W(k+1))**T */
                    if (k < n - 2) {
                        c64 d21 = A[(k + 1) + k * lda];
                        c64 d11 = A[(k + 1) + (k + 1) * lda] / d21;
                        c64 d22 = A[k + k * lda] / d21;
                        c64 t = CONE / (d11 * d22 - CONE);
                        d21 = t / d21;

                        for (INT j = k + 2; j < n; j++) {
                            c64 wk = d21 * (d11 * A[j + k * lda] - A[j + (k + 1) * lda]);
                            c64 wkp1 = d21 * (d22 * A[j + (k + 1) * lda] - A[j + k * lda]);
                            for (INT i = j; i < n; i++) {
                                A[i + j * lda] -= A[i + k * lda] * wk +
                                                  A[i + (k + 1) * lda] * wkp1;
                            }
                            A[j + k * lda] = wk;
                            A[j + (k + 1) * lda] = wkp1;
                        }
                    }
                }

                /* Store details of the interchanges in ipiv */
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
