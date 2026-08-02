/**
 * @file chetf2_rk.c
 * @brief CHETF2_RK computes the factorization of a complex Hermitian indefinite matrix using the bounded Bunch-Kaufman (rook) diagonal pivoting method (BLAS2 unblocked algorithm).
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CHETF2_RK computes the factorization of a complex Hermitian matrix A
 * using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
 * @rst
 * .. code-block:: text
 *
 *     A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
 * @endrst
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**H (or L**H) is the conjugate transpose of U (or L), P is a
 * permutation matrix, P**T is the transpose of P, and D is Hermitian
 * and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular
 *                       - `'L'`: Lower triangular
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *                      On entry, the Hermitian matrix A.
 *                      On exit, contains:
 *                      - Only diagonal elements of the Hermitian block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D are stored on exit in array `E`), and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[out]    E     Array of dimension `n`.
 *                      On exit, contains the superdiagonal (or subdiagonal)
 *                      elements of the Hermitian block diagonal matrix D
 *                      with 1-by-1 or 2-by-2 diagonal blocks, where
 *                      if `uplo='U'`: `E[i]=D(i-1,i)`, `i=1:n-1`, `E[0]` is
 *                      set to 0; if `uplo='L'`: `E[i]=D(i+1,i)`, `i=0:n-2`,
 *                      `E[n-1]` is set to 0.
 *                      For a 1-by-1 diagonal block `D(k)`, the element `E[k]`
 *                      is set to 0 in both `uplo='U'` and `uplo='L'` cases.
 * @param[out]    ipiv  Array of dimension `n`. Pivot indices (0-based).
 *                      If `uplo='U'`:
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
void chetf2_rk(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    c64* restrict E,
    INT* restrict ipiv,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 EIGHT = 8.0f;
    const f32 SEVTEN = 17.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    INT upper, done;
    INT i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f32 absakk, alpha, colmax, d11, d22, rowmax, dtemp, tt, d, r1, sfmin;
    c64 d12, d21, t, wk, wkm1, wkp1;

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
        xerbla("CHETF2_RK", -(*info));
        return;
    }

    alpha = (ONE + sqrtf(SEVTEN)) / EIGHT;

    sfmin = slamch("S");

    if (upper) {

        E[0] = CZERO;

        k = n - 1;

        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = fabsf(crealf(A[k + k * lda]));

            if (k > 0) {
                imax = cblas_icamax(k, &A[0 + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            } else {
                colmax = ZERO;
            }

            if (fmaxf(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = crealf(A[k + k * lda]);

                if (k > 0) {
                    E[k] = CZERO;
                }

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax != k) {
                            jmax = imax + 1 + cblas_icamax(k - imax, &A[imax + (imax + 1) * lda], lda);
                            rowmax = cabs1f(A[imax + jmax * lda]);
                        } else {
                            rowmax = ZERO;
                        }

                        if (imax > 0) {
                            itemp = cblas_icamax(imax, &A[0 + imax * lda], 1);
                            dtemp = cabs1f(A[itemp + imax * lda]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabsf(crealf(A[imax + imax * lda])) < alpha * rowmax)) {

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
                        cblas_cswap(p, &A[0 + k * lda], 1, &A[0 + p * lda], 1);
                    }
                    for (j = p + 1; j <= k - 1; j++) {
                        t = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[p + j * lda]);
                        A[p + j * lda] = t;
                    }
                    A[p + k * lda] = conjf(A[p + k * lda]);
                    r1 = crealf(A[k + k * lda]);
                    A[k + k * lda] = crealf(A[p + p * lda]);
                    A[p + p * lda] = r1;

                    if (k < n - 1) {
                        cblas_cswap(n - k - 1, &A[k + (k + 1) * lda], lda, &A[p + (k + 1) * lda], lda);
                    }
                }

                kk = k - kstep + 1;
                if (kp != kk) {

                    if (kp > 0) {
                        cblas_cswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    for (j = kp + 1; j <= kk - 1; j++) {
                        t = conjf(A[j + kk * lda]);
                        A[j + kk * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = t;
                    }
                    A[kp + kk * lda] = conjf(A[kp + kk * lda]);
                    r1 = crealf(A[kk + kk * lda]);
                    A[kk + kk * lda] = crealf(A[kp + kp * lda]);
                    A[kp + kp * lda] = r1;
                    if (kstep == 2) {
                        A[k + k * lda] = crealf(A[k + k * lda]);
                        t = A[k - 1 + k * lda];
                        A[k - 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }

                    if (k < n - 1) {
                        cblas_cswap(n - k - 1, &A[kk + (k + 1) * lda], lda, &A[kp + (k + 1) * lda], lda);
                    }

                } else {
                    A[k + k * lda] = crealf(A[k + k * lda]);
                    if (kstep == 2) {
                        A[k - 1 + (k - 1) * lda] = crealf(A[k - 1 + (k - 1) * lda]);
                    }
                }

                if (kstep == 1) {

                    if (k > 0) {

                        if (fabsf(crealf(A[k + k * lda])) >= sfmin) {

                            d11 = ONE / crealf(A[k + k * lda]);
                            cblas_cher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_csscal(k, d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = crealf(A[k + k * lda]);
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_cher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }

                        E[k] = CZERO;
                    }

                } else {

                    if (k > 1) {

                        d = slapy2(crealf(A[k - 1 + k * lda]), cimagf(A[k - 1 + k * lda]));
                        d11 = crealf(A[k + k * lda]) / d;
                        d22 = crealf(A[k - 1 + (k - 1) * lda]) / d;
                        d12 = A[k - 1 + k * lda] / d;
                        tt = ONE / (d11 * d22 - ONE);

                        for (j = k - 2; j >= 0; j--) {

                            wkm1 = tt * (d11 * A[j + (k - 1) * lda] - conjf(d12) * A[j + k * lda]);
                            wk = tt * (d22 * A[j + k * lda] - d12 * A[j + (k - 1) * lda]);

                            for (i = j; i >= 0; i--) {
                                A[i + j * lda] = A[i + j * lda] -
                                                 (A[i + k * lda] / d) * conjf(wk) -
                                                 (A[i + (k - 1) * lda] / d) * conjf(wkm1);
                            }

                            A[j + k * lda] = wk / d;
                            A[j + (k - 1) * lda] = wkm1 / d;
                            A[j + j * lda] = CMPLXF(crealf(A[j + j * lda]), 0.0f);
                        }
                    }

                    E[k] = A[k - 1 + k * lda];
                    E[k - 1] = CZERO;
                    A[k - 1 + k * lda] = CZERO;

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

        E[n - 1] = CZERO;

        k = 0;

        while (k < n) {

            kstep = 1;
            p = k;

            absakk = fabsf(crealf(A[k + k * lda]));

            if (k < n - 1) {
                imax = k + 1 + cblas_icamax(n - k - 1, &A[k + 1 + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            } else {
                colmax = ZERO;
            }

            if (fmaxf(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = crealf(A[k + k * lda]);

                if (k < n - 1) {
                    E[k] = CZERO;
                }

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax != k) {
                            jmax = k + cblas_icamax(imax - k, &A[imax + k * lda], lda);
                            rowmax = cabs1f(A[imax + jmax * lda]);
                        } else {
                            rowmax = ZERO;
                        }

                        if (imax < n - 1) {
                            itemp = imax + 1 + cblas_icamax(n - imax - 1, &A[imax + 1 + imax * lda], 1);
                            dtemp = cabs1f(A[itemp + imax * lda]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabsf(crealf(A[imax + imax * lda])) < alpha * rowmax)) {

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
                        cblas_cswap(n - p - 1, &A[p + 1 + k * lda], 1, &A[p + 1 + p * lda], 1);
                    }
                    for (j = k + 1; j <= p - 1; j++) {
                        t = conjf(A[j + k * lda]);
                        A[j + k * lda] = conjf(A[p + j * lda]);
                        A[p + j * lda] = t;
                    }
                    A[p + k * lda] = conjf(A[p + k * lda]);
                    r1 = crealf(A[k + k * lda]);
                    A[k + k * lda] = crealf(A[p + p * lda]);
                    A[p + p * lda] = r1;

                    if (k > 0) {
                        cblas_cswap(k, &A[k + 0 * lda], lda, &A[p + 0 * lda], lda);
                    }
                }

                kk = k + kstep - 1;
                if (kp != kk) {

                    if (kp < n - 1) {
                        cblas_cswap(n - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    for (j = kk + 1; j <= kp - 1; j++) {
                        t = conjf(A[j + kk * lda]);
                        A[j + kk * lda] = conjf(A[kp + j * lda]);
                        A[kp + j * lda] = t;
                    }
                    A[kp + kk * lda] = conjf(A[kp + kk * lda]);
                    r1 = crealf(A[kk + kk * lda]);
                    A[kk + kk * lda] = crealf(A[kp + kp * lda]);
                    A[kp + kp * lda] = r1;
                    if (kstep == 2) {
                        A[k + k * lda] = crealf(A[k + k * lda]);
                        t = A[k + 1 + k * lda];
                        A[k + 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }

                    if (k > 0) {
                        cblas_cswap(k, &A[kk + 0 * lda], lda, &A[kp + 0 * lda], lda);
                    }

                } else {
                    A[k + k * lda] = crealf(A[k + k * lda]);
                    if (kstep == 2) {
                        A[k + 1 + (k + 1) * lda] = crealf(A[k + 1 + (k + 1) * lda]);
                    }
                }

                if (kstep == 1) {

                    if (k < n - 1) {

                        if (fabsf(crealf(A[k + k * lda])) >= sfmin) {

                            d11 = ONE / crealf(A[k + k * lda]);
                            cblas_cher(CblasColMajor, CblasLower, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);

                            cblas_csscal(n - k - 1, d11, &A[k + 1 + k * lda], 1);

                        } else {

                            d11 = crealf(A[k + k * lda]);
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_cher(CblasColMajor, CblasLower, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);
                        }

                        E[k] = CZERO;
                    }

                } else {

                    if (k < n - 2) {

                        d = slapy2(crealf(A[k + 1 + k * lda]), cimagf(A[k + 1 + k * lda]));
                        d11 = crealf(A[k + 1 + (k + 1) * lda]) / d;
                        d22 = crealf(A[k + k * lda]) / d;
                        d21 = A[k + 1 + k * lda] / d;
                        tt = ONE / (d11 * d22 - ONE);

                        for (j = k + 2; j < n; j++) {

                            wk = tt * (d11 * A[j + k * lda] - d21 * A[j + (k + 1) * lda]);
                            wkp1 = tt * (d22 * A[j + (k + 1) * lda] - conjf(d21) * A[j + k * lda]);

                            for (i = j; i < n; i++) {
                                A[i + j * lda] = A[i + j * lda] -
                                                 (A[i + k * lda] / d) * conjf(wk) -
                                                 (A[i + (k + 1) * lda] / d) * conjf(wkp1);
                            }

                            A[j + k * lda] = wk / d;
                            A[j + (k + 1) * lda] = wkp1 / d;
                            A[j + j * lda] = CMPLXF(crealf(A[j + j * lda]), 0.0f);
                        }
                    }

                    E[k] = A[k + 1 + k * lda];
                    E[k + 1] = CZERO;
                    A[k + 1 + k * lda] = CZERO;

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
