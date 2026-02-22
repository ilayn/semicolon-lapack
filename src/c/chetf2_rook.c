/**
 * @file chetf2_rook.c
 * @brief CHETF2_ROOK computes the factorization of a complex Hermitian indefinite matrix using the bounded Bunch-Kaufman ("rook") diagonal pivoting method (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETF2_ROOK computes the factorization of a complex Hermitian matrix A
 * using the bounded Bunch-Kaufman ("rook") diagonal pivoting method:
 *
 *    A = U*D*U**H  or  A = L*D*L**H
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, U**H is the conjugate transpose of U, and D is
 * Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Single complex array, dimension (lda, n).
 *          On entry, the Hermitian matrix A.
 *          On exit, the block diagonal matrix D and the multipliers.
 *
 * @param[in] lda
 *          The leading dimension of A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and block structure.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, D(k,k) is exactly zero.
 */
void chetf2_rook(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    INT upper, done;
    INT i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f32 absakk, alpha, colmax, d, d11, d22, r1, dtemp, rowmax, tt, sfmin;
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
        xerbla("CHETF2_ROOK", -(*info));
        return;
    }

    alpha = (1.0f + sqrtf(17.0f)) / 8.0f;

    sfmin = slamch("S");

    if (upper) {

        k = n - 1;
        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = fabsf(crealf(A[k + k * lda]));

            if (k > 0) {
                imax = cblas_icamax(k, &A[0 + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            } else {
                colmax = 0.0f;
            }

            if (fmaxf(absakk, colmax) == 0.0f) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = crealf(A[k + k * lda]);

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
                            rowmax = 0.0f;
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
                } else {
                    A[k + k * lda] = crealf(A[k + k * lda]);
                    if (kstep == 2) {
                        A[k - 1 + (k - 1) * lda] = crealf(A[k - 1 + (k - 1) * lda]);
                    }
                }

                if (kstep == 1) {

                    if (k > 0) {

                        if (fabsf(crealf(A[k + k * lda])) >= sfmin) {

                            d11 = 1.0f / crealf(A[k + k * lda]);
                            cblas_cher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_csscal(k, d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = crealf(A[k + k * lda]);
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_cher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }
                    }

                } else {

                    if (k > 1) {

                        d = slapy2(crealf(A[k - 1 + k * lda]), cimagf(A[k - 1 + k * lda]));
                        d11 = crealf(A[k + k * lda]) / d;
                        d22 = crealf(A[k - 1 + (k - 1) * lda]) / d;
                        d12 = A[k - 1 + k * lda] / d;
                        tt = 1.0f / (d11 * d22 - 1.0f);

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

            absakk = fabsf(crealf(A[k + k * lda]));

            if (k < n - 1) {
                imax = k + 1 + cblas_icamax(n - k - 1, &A[k + 1 + k * lda], 1);
                colmax = cabs1f(A[imax + k * lda]);
            } else {
                colmax = 0.0f;
            }

            if (fmaxf(absakk, colmax) == 0.0f) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = crealf(A[k + k * lda]);

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
                            rowmax = 0.0f;
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
                } else {
                    A[k + k * lda] = crealf(A[k + k * lda]);
                    if (kstep == 2) {
                        A[k + 1 + (k + 1) * lda] = crealf(A[k + 1 + (k + 1) * lda]);
                    }
                }

                if (kstep == 1) {

                    if (k < n - 1) {

                        if (fabsf(crealf(A[k + k * lda])) >= sfmin) {

                            d11 = 1.0f / crealf(A[k + k * lda]);
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
                    }

                } else {

                    if (k < n - 2) {

                        d = slapy2(crealf(A[k + 1 + k * lda]), cimagf(A[k + 1 + k * lda]));
                        d11 = crealf(A[k + 1 + (k + 1) * lda]) / d;
                        d22 = crealf(A[k + k * lda]) / d;
                        d21 = A[k + 1 + k * lda] / d;
                        tt = 1.0f / (d11 * d22 - 1.0f);

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
