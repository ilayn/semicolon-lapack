/**
 * @file ssytf2_rk.c
 * @brief SSYTF2_RK computes the factorization of a real symmetric indefinite matrix using the bounded Bunch-Kaufman (rook) diagonal pivoting method (BLAS2 unblocked algorithm).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SSYTF2_RK computes the factorization of a real symmetric matrix A
 * using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
 * @rst
 * .. code-block:: text
 *
 *     A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 * @endrst
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular
 *                       - `'L'`: Lower triangular
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *                      On entry, the symmetric matrix A.
 *                      On exit, contains:
 *
 *                      - Only diagonal elements of the symmetric block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D are stored on exit in array `E`), and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[out]    E     Array of dimension `n`.
 *                      On exit, contains the superdiagonal (or subdiagonal)
 *                      elements of the symmetric block diagonal matrix D
 *                      with 1-by-1 or 2-by-2 diagonal blocks, where
 *                      if `uplo='U'`: `E[i]=D(i-1,i)`, `i=1:n-1`, `E[0]` is
 *                      set to 0; if `uplo='L'`: `E[i]=D(i+1,i)`, `i=0:n-2`,
 *                      `E[n-1]` is set to 0.
 *                      For a 1-by-1 diagonal block `D(k)`, the element `E[k]`
 *                      is set to 0 in both `uplo='U'` and `uplo='L'` cases.
 * @param[out]    ipiv  Array of dimension `n`. Pivot indices (0-based).
 *                      If `uplo='U'`:
 *
 *                      - If `ipiv[k]>=0`, then rows and columns `k` and
 *                        `ipiv[k]` were interchanged and `D(k,k)` is a
 *                        1-by-1 diagonal block.
 *                      - If `ipiv[k]<0` and `ipiv[k-1]<0`, then rows and
 *                        columns `k` and `-ipiv[k]-1` were interchanged and
 *                        rows and columns `k-1` and `-ipiv[k-1]-1` were
 *                        interchanged, `D(k-1:k,k-1:k)` is a 2-by-2
 *                        diagonal block.
 *
 *                      If `uplo='L'`:
 *
 *                      - If `ipiv[k]>=0`, then rows and columns `k` and
 *                        `ipiv[k]` were interchanged and `D(k,k)` is a
 *                        1-by-1 diagonal block.
 *                      - If `ipiv[k]<0` and `ipiv[k+1]<0`, then rows and
 *                        columns `k` and `-ipiv[k]-1` were interchanged and
 *                        rows and columns `k+1` and `-ipiv[k+1]-1` were
 *                        interchanged, `D(k:k+1,k:k+1)` is a 2-by-2
 *                        diagonal block.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-k`, the k-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=k`, the matrix A is singular, because
 *                           column k of the triangular part of A (upper or lower,
 *                           per `uplo`) contains all zeros. `D(k,k)` is exactly
 *                           zero, and superdiagonal (or subdiagonal) elements of
 *                           column k of U (or L) are all zeros. The factorization
 *                           has been completed, but the block diagonal matrix D is
 *                           exactly singular, and division by zero will occur if it
 *                           is used to solve a system of equations. `info` only
 *                           stores the first occurrence of a singularity; the
 *                           factorization always completes.
 */
void ssytf2_rk(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    f32* restrict E,
    INT* restrict ipiv,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 EIGHT = 8.0f;
    const f32 SEVTEN = 17.0f;

    INT upper, done;
    INT i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f32 absakk, alpha, colmax, d11, d12, d21, d22, rowmax, dtemp, t, wk, wkm1, wkp1, sfmin;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SSYTF2_RK", -(*info));
        return;
    }

    alpha = (ONE + sqrtf(SEVTEN)) / EIGHT;

    sfmin = slamch("S");

    if (upper) {

        E[0] = ZERO;

        k = n - 1;

        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = fabsf(A[k + k * lda]);

            if (k > 0) {
                imax = cblas_isamax(k, &A[0 + k * lda], 1);
                colmax = fabsf(A[imax + k * lda]);
            } else {
                colmax = ZERO;
            }

            if (fmaxf(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;

                if (k > 0) {
                    E[k] = ZERO;
                }

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax != k) {
                            jmax = imax + 1 + cblas_isamax(k - imax, &A[imax + (imax + 1) * lda], lda);
                            rowmax = fabsf(A[imax + jmax * lda]);
                        } else {
                            rowmax = ZERO;
                        }

                        if (imax > 0) {
                            itemp = cblas_isamax(imax, &A[0 + imax * lda], 1);
                            dtemp = fabsf(A[itemp + imax * lda]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabsf(A[imax + imax * lda]) < alpha * rowmax)) {

                            kp = imax;
                            done = 1;

                        } else if ((p == jmax) || (rowmax <= colmax)) {

                            kp = imax;
                            kstep = 2;
                            done = 1;

                        } else {

                            p = imax;
                            colmax = rowmax;
                            imax = jmax;

                        }
                    }
                }

                if ((kstep == 2) && (p != k)) {

                    if (p > 0) {
                        cblas_sswap(p, &A[0 + k * lda], 1, &A[0 + p * lda], 1);
                    }
                    if (p < k - 1) {
                        cblas_sswap(k - p - 1, &A[p + 1 + k * lda], 1, &A[p + (p + 1) * lda], lda);
                    }
                    t = A[k + k * lda];
                    A[k + k * lda] = A[p + p * lda];
                    A[p + p * lda] = t;

                    if (k < n - 1) {
                        cblas_sswap(n - k - 1, &A[k + (k + 1) * lda], lda, &A[p + (k + 1) * lda], lda);
                    }
                }

                kk = k - kstep + 1;
                if (kp != kk) {

                    if (kp > 0) {
                        cblas_sswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    if ((kk > 0) && (kp < kk - 1)) {
                        cblas_sswap(kk - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    }
                    t = A[kk + kk * lda];
                    A[kk + kk * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = t;
                    if (kstep == 2) {
                        t = A[k - 1 + k * lda];
                        A[k - 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }

                    if (k < n - 1) {
                        cblas_sswap(n - k - 1, &A[kk + (k + 1) * lda], lda, &A[kp + (k + 1) * lda], lda);
                    }
                }

                if (kstep == 1) {

                    if (k > 0) {

                        if (fabsf(A[k + k * lda]) >= sfmin) {

                            d11 = ONE / A[k + k * lda];
                            cblas_ssyr(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_sscal(k, d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = A[k + k * lda];
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_ssyr(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }

                        E[k] = ZERO;
                    }

                } else {

                    if (k > 1) {

                        d12 = A[k - 1 + k * lda];
                        d22 = A[k - 1 + (k - 1) * lda] / d12;
                        d11 = A[k + k * lda] / d12;
                        t = ONE / (d11 * d22 - ONE);

                        for (j = k - 2; j >= 0; j--) {

                            wkm1 = t * (d11 * A[j + (k - 1) * lda] - A[j + k * lda]);
                            wk = t * (d22 * A[j + k * lda] - A[j + (k - 1) * lda]);

                            for (i = j; i >= 0; i--) {
                                A[i + j * lda] = A[i + j * lda] - (A[i + k * lda] / d12) * wk -
                                                 (A[i + (k - 1) * lda] / d12) * wkm1;
                            }

                            A[j + k * lda] = wk / d12;
                            A[j + (k - 1) * lda] = wkm1 / d12;
                        }
                    }

                    E[k] = A[k - 1 + k * lda];
                    E[k - 1] = ZERO;
                    A[k - 1 + k * lda] = ZERO;

                }
            }

            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(p + 1);
                ipiv[k - 1] = -(kp + 1);
            }

            k = k - kstep;
        }

    } else {

        E[n - 1] = ZERO;

        k = 0;

        while (k < n) {

            kstep = 1;
            p = k;

            absakk = fabsf(A[k + k * lda]);

            if (k < n - 1) {
                imax = k + 1 + cblas_isamax(n - k - 1, &A[k + 1 + k * lda], 1);
                colmax = fabsf(A[imax + k * lda]);
            } else {
                colmax = ZERO;
            }

            if (fmaxf(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;

                if (k < n - 1) {
                    E[k] = ZERO;
                }

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax != k) {
                            jmax = k + cblas_isamax(imax - k, &A[imax + k * lda], lda);
                            rowmax = fabsf(A[imax + jmax * lda]);
                        } else {
                            rowmax = ZERO;
                        }

                        if (imax < n - 1) {
                            itemp = imax + 1 + cblas_isamax(n - imax - 1, &A[imax + 1 + imax * lda], 1);
                            dtemp = fabsf(A[itemp + imax * lda]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabsf(A[imax + imax * lda]) < alpha * rowmax)) {

                            kp = imax;
                            done = 1;

                        } else if ((p == jmax) || (rowmax <= colmax)) {

                            kp = imax;
                            kstep = 2;
                            done = 1;

                        } else {

                            p = imax;
                            colmax = rowmax;
                            imax = jmax;

                        }
                    }
                }

                if ((kstep == 2) && (p != k)) {

                    if (p < n - 1) {
                        cblas_sswap(n - p - 1, &A[p + 1 + k * lda], 1, &A[p + 1 + p * lda], 1);
                    }
                    if (p > k + 1) {
                        cblas_sswap(p - k - 1, &A[k + 1 + k * lda], 1, &A[p + (k + 1) * lda], lda);
                    }
                    t = A[k + k * lda];
                    A[k + k * lda] = A[p + p * lda];
                    A[p + p * lda] = t;

                    if (k > 0) {
                        cblas_sswap(k, &A[k + 0 * lda], lda, &A[p + 0 * lda], lda);
                    }
                }

                kk = k + kstep - 1;
                if (kp != kk) {

                    if (kp < n - 1) {
                        cblas_sswap(n - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    if ((kk < n - 1) && (kp > kk + 1)) {
                        cblas_sswap(kp - kk - 1, &A[kk + 1 + kk * lda], 1, &A[kp + (kk + 1) * lda], lda);
                    }
                    t = A[kk + kk * lda];
                    A[kk + kk * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = t;
                    if (kstep == 2) {
                        t = A[k + 1 + k * lda];
                        A[k + 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }

                    if (k > 0) {
                        cblas_sswap(k, &A[kk + 0 * lda], lda, &A[kp + 0 * lda], lda);
                    }
                }

                if (kstep == 1) {

                    if (k < n - 1) {

                        if (fabsf(A[k + k * lda]) >= sfmin) {

                            d11 = ONE / A[k + k * lda];
                            cblas_ssyr(CblasColMajor, CblasLower, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);

                            cblas_sscal(n - k - 1, d11, &A[k + 1 + k * lda], 1);

                        } else {

                            d11 = A[k + k * lda];
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_ssyr(CblasColMajor, CblasLower, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);
                        }

                        E[k] = ZERO;
                    }

                } else {

                    if (k < n - 2) {

                        d21 = A[k + 1 + k * lda];
                        d11 = A[k + 1 + (k + 1) * lda] / d21;
                        d22 = A[k + k * lda] / d21;
                        t = ONE / (d11 * d22 - ONE);

                        for (j = k + 2; j < n; j++) {

                            wk = t * (d11 * A[j + k * lda] - A[j + (k + 1) * lda]);
                            wkp1 = t * (d22 * A[j + (k + 1) * lda] - A[j + k * lda]);

                            for (i = j; i < n; i++) {
                                A[i + j * lda] = A[i + j * lda] - (A[i + k * lda] / d21) * wk -
                                                 (A[i + (k + 1) * lda] / d21) * wkp1;
                            }

                            A[j + k * lda] = wk / d21;
                            A[j + (k + 1) * lda] = wkp1 / d21;
                        }
                    }

                    E[k] = A[k + 1 + k * lda];
                    E[k + 1] = ZERO;
                    A[k + 1 + k * lda] = ZERO;

                }
            }

            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(p + 1);
                ipiv[k + 1] = -(kp + 1);
            }

            k = k + kstep;
        }
    }
}
