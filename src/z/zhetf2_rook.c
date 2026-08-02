/**
 * @file zhetf2_rook.c
 * @brief ZHETF2_ROOK computes the factorization of a complex Hermitian indefinite matrix using the bounded Bunch-Kaufman ("rook") diagonal pivoting method (unblocked algorithm).
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZHETF2_ROOK computes the factorization of a complex Hermitian matrix A
 * using the bounded Bunch-Kaufman ("rook") diagonal pivoting method:
 * @rst
 * .. code-block:: text
 *
 *     A = U*D*U**H  or  A = L*D*L**H
 * @endrst
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, U**H is the conjugate transpose of U, and D is
 * Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular
 *                       - `'L'`: Lower triangular
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *                      On entry, the Hermitian matrix A.
 *                      On exit, the block diagonal matrix D and the multipliers
 *                      (see below for further details).
 * @param[in]     lda   The leading dimension of A. `lda>=max(1,n)`.
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
 *                         - `info>0`: if `info=k`, `D(k,k)` is exactly zero. The
 *                           factorization has been completed, but D is exactly
 *                           singular, and division by zero will occur if it is
 *                           used to solve a system of equations.
 *
 * @par Further Details:
 * @rst
 * If ``uplo='U'``, then A = U*D*U**H, where
 * U = P(n-1)*U(n-1)* ... *P(k)*U(k)* ..., i.e., U is a product of terms
 * P(k)*U(k), where k decreases from n-1 to 0 in steps of 1 or 2, and D
 * is a block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks
 * D(k). P(k) is a permutation matrix as defined by ``ipiv[k]``, and
 * U(k) is a unit upper triangular matrix, such that if the diagonal
 * block D(k) is of order s (s = 1 or 2), then
 *
 * .. code-block:: text
 *
 *              (   I    v    0   )   k-s+1
 *      U(k) =  (   0    I    0   )   s
 *              (   0    0    I   )   n-1-k
 *                 k-s+1  s   n-1-k
 *
 * If s = 1, D(k) overwrites A(k,k), and v overwrites A(0:k-1,k).
 * If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
 * and A(k,k), and v overwrites A(0:k-2,k-1:k).
 *
 * If ``uplo='L'``, then A = L*D*L**H, where
 * L = P(0)*L(0)* ... *P(k)*L(k)* ..., i.e., L is a product of terms
 * P(k)*L(k), where k increases from 0 to n-1 in steps of 1 or 2, and D
 * is a block diagonal matrix with 1-by-1 and 2-by-2 diagonal blocks
 * D(k). P(k) is a permutation matrix as defined by ``ipiv[k]``, and
 * L(k) is a unit lower triangular matrix, such that if the diagonal
 * block D(k) is of order s (s = 1 or 2), then
 *
 * .. code-block:: text
 *
 *              (   I    0     0   )  k
 *      L(k) =  (   0    I     0   )  s
 *              (   0    v     I   )  n-k-s
 *                 k     s   n-k-s
 *
 * If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n-1,k).
 * If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
 * and A(k+1,k+1), and v overwrites A(k+2:n-1,k:k+1).
 * @endrst
 */
void zhetf2_rook(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    INT upper, done;
    INT i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f64 absakk, alpha, colmax, d, d11, d22, r1, dtemp, rowmax, tt, sfmin;
    c128 d12, d21, t, wk, wkm1, wkp1;

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
        xerbla("ZHETF2_ROOK", -(*info));
        return;
    }

    alpha = (1.0 + sqrt(17.0)) / 8.0;

    sfmin = dlamch("S");

    if (upper) {

        k = n - 1;
        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = fabs(creal(A[k + k * lda]));

            if (k > 0) {
                imax = cblas_izamax(k, &A[0 + k * lda], 1);
                colmax = cabs1(A[imax + k * lda]);
            } else {
                colmax = 0.0;
            }

            if (fmax(absakk, colmax) == 0.0) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(A[k + k * lda]);

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax != k) {
                            jmax = imax + 1 + cblas_izamax(k - imax, &A[imax + (imax + 1) * lda], lda);
                            rowmax = cabs1(A[imax + jmax * lda]);
                        } else {
                            rowmax = 0.0;
                        }

                        if (imax > 0) {
                            itemp = cblas_izamax(imax, &A[0 + imax * lda], 1);
                            dtemp = cabs1(A[itemp + imax * lda]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabs(creal(A[imax + imax * lda])) < alpha * rowmax)) {

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
                        cblas_zswap(p, &A[0 + k * lda], 1, &A[0 + p * lda], 1);
                    }
                    for (j = p + 1; j <= k - 1; j++) {
                        t = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[p + j * lda]);
                        A[p + j * lda] = t;
                    }
                    A[p + k * lda] = conj(A[p + k * lda]);
                    r1 = creal(A[k + k * lda]);
                    A[k + k * lda] = creal(A[p + p * lda]);
                    A[p + p * lda] = r1;
                }

                kk = k - kstep + 1;
                if (kp != kk) {
                    if (kp > 0) {
                        cblas_zswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    for (j = kp + 1; j <= kk - 1; j++) {
                        t = conj(A[j + kk * lda]);
                        A[j + kk * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = t;
                    }
                    A[kp + kk * lda] = conj(A[kp + kk * lda]);
                    r1 = creal(A[kk + kk * lda]);
                    A[kk + kk * lda] = creal(A[kp + kp * lda]);
                    A[kp + kp * lda] = r1;
                    if (kstep == 2) {
                        A[k + k * lda] = creal(A[k + k * lda]);
                        t = A[k - 1 + k * lda];
                        A[k - 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                } else {
                    A[k + k * lda] = creal(A[k + k * lda]);
                    if (kstep == 2) {
                        A[k - 1 + (k - 1) * lda] = creal(A[k - 1 + (k - 1) * lda]);
                    }
                }

                if (kstep == 1) {

                    if (k > 0) {

                        if (fabs(creal(A[k + k * lda])) >= sfmin) {

                            d11 = 1.0 / creal(A[k + k * lda]);
                            cblas_zher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_zdscal(k, d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = creal(A[k + k * lda]);
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_zher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }
                    }

                } else {

                    if (k > 1) {

                        d = dlapy2(creal(A[k - 1 + k * lda]), cimag(A[k - 1 + k * lda]));
                        d11 = creal(A[k + k * lda]) / d;
                        d22 = creal(A[k - 1 + (k - 1) * lda]) / d;
                        d12 = A[k - 1 + k * lda] / d;
                        tt = 1.0 / (d11 * d22 - 1.0);

                        for (j = k - 2; j >= 0; j--) {

                            wkm1 = tt * (d11 * A[j + (k - 1) * lda] - conj(d12) * A[j + k * lda]);
                            wk = tt * (d22 * A[j + k * lda] - d12 * A[j + (k - 1) * lda]);

                            for (i = j; i >= 0; i--) {
                                A[i + j * lda] = A[i + j * lda] -
                                                 (A[i + k * lda] / d) * conj(wk) -
                                                 (A[i + (k - 1) * lda] / d) * conj(wkm1);
                            }

                            A[j + k * lda] = wk / d;
                            A[j + (k - 1) * lda] = wkm1 / d;
                            A[j + j * lda] = CMPLX(creal(A[j + j * lda]), 0.0);
                        }
                    }
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

        k = 0;
        while (k < n) {

            kstep = 1;
            p = k;

            absakk = fabs(creal(A[k + k * lda]));

            if (k < n - 1) {
                imax = k + 1 + cblas_izamax(n - k - 1, &A[k + 1 + k * lda], 1);
                colmax = cabs1(A[imax + k * lda]);
            } else {
                colmax = 0.0;
            }

            if (fmax(absakk, colmax) == 0.0) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(A[k + k * lda]);

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax != k) {
                            jmax = k + cblas_izamax(imax - k, &A[imax + k * lda], lda);
                            rowmax = cabs1(A[imax + jmax * lda]);
                        } else {
                            rowmax = 0.0;
                        }

                        if (imax < n - 1) {
                            itemp = imax + 1 + cblas_izamax(n - imax - 1, &A[imax + 1 + imax * lda], 1);
                            dtemp = cabs1(A[itemp + imax * lda]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabs(creal(A[imax + imax * lda])) < alpha * rowmax)) {

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
                        cblas_zswap(n - p - 1, &A[p + 1 + k * lda], 1, &A[p + 1 + p * lda], 1);
                    }
                    for (j = k + 1; j <= p - 1; j++) {
                        t = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[p + j * lda]);
                        A[p + j * lda] = t;
                    }
                    A[p + k * lda] = conj(A[p + k * lda]);
                    r1 = creal(A[k + k * lda]);
                    A[k + k * lda] = creal(A[p + p * lda]);
                    A[p + p * lda] = r1;
                }

                kk = k + kstep - 1;
                if (kp != kk) {
                    if (kp < n - 1) {
                        cblas_zswap(n - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    for (j = kk + 1; j <= kp - 1; j++) {
                        t = conj(A[j + kk * lda]);
                        A[j + kk * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = t;
                    }
                    A[kp + kk * lda] = conj(A[kp + kk * lda]);
                    r1 = creal(A[kk + kk * lda]);
                    A[kk + kk * lda] = creal(A[kp + kp * lda]);
                    A[kp + kp * lda] = r1;
                    if (kstep == 2) {
                        A[k + k * lda] = creal(A[k + k * lda]);
                        t = A[k + 1 + k * lda];
                        A[k + 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                } else {
                    A[k + k * lda] = creal(A[k + k * lda]);
                    if (kstep == 2) {
                        A[k + 1 + (k + 1) * lda] = creal(A[k + 1 + (k + 1) * lda]);
                    }
                }

                if (kstep == 1) {

                    if (k < n - 1) {

                        if (fabs(creal(A[k + k * lda])) >= sfmin) {

                            d11 = 1.0 / creal(A[k + k * lda]);
                            cblas_zher(CblasColMajor, CblasLower, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);

                            cblas_zdscal(n - k - 1, d11, &A[k + 1 + k * lda], 1);

                        } else {

                            d11 = creal(A[k + k * lda]);
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_zher(CblasColMajor, CblasLower, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);
                        }
                    }

                } else {

                    if (k < n - 2) {

                        d = dlapy2(creal(A[k + 1 + k * lda]), cimag(A[k + 1 + k * lda]));
                        d11 = creal(A[k + 1 + (k + 1) * lda]) / d;
                        d22 = creal(A[k + k * lda]) / d;
                        d21 = A[k + 1 + k * lda] / d;
                        tt = 1.0 / (d11 * d22 - 1.0);

                        for (j = k + 2; j < n; j++) {

                            wk = tt * (d11 * A[j + k * lda] - d21 * A[j + (k + 1) * lda]);
                            wkp1 = tt * (d22 * A[j + (k + 1) * lda] - conj(d21) * A[j + k * lda]);

                            for (i = j; i < n; i++) {
                                A[i + j * lda] = A[i + j * lda] -
                                                 (A[i + k * lda] / d) * conj(wk) -
                                                 (A[i + (k + 1) * lda] / d) * conj(wkp1);
                            }

                            A[j + k * lda] = wk / d;
                            A[j + (k + 1) * lda] = wkp1 / d;
                            A[j + j * lda] = CMPLX(creal(A[j + j * lda]), 0.0);
                        }
                    }
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
