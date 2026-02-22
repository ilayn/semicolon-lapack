/**
 * @file csytf2_rook.c
 * @brief CSYTF2_ROOK computes the factorization of a complex symmetric indefinite matrix using the bounded Bunch-Kaufman ("rook") diagonal pivoting method (unblocked algorithm).
 */

#include "semicolon_lapack_complex_single.h"
#include "semicolon_cblas.h"
#include <complex.h>
#include <math.h>

/**
 * CSYTF2_ROOK computes the factorization of a complex symmetric matrix A
 * using the bounded Bunch-Kaufman ("rook") diagonal pivoting method:
 *
 *    A = U*D*U**T  or  A = L*D*L**T
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
 *          On entry, the symmetric matrix A.
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
void csytf2_rook(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT upper, done;
    INT i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f32 absakk, alpha, colmax, rowmax, dtemp, sfmin;
    c64 d11, d12, d21, d22, t, wk, wkm1, wkp1;

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
        xerbla("CSYTF2_ROOK", -(*info));
        return;
    }

    alpha = (1.0f + sqrtf(17.0f)) / 8.0f;

    sfmin = slamch("S");

    if (upper) {

        k = n - 1;
        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = cabs1f(A[k + k * lda]);

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

                        if (!(cabs1f(A[imax + imax * lda]) < alpha * rowmax)) {

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
                    if (p < k - 1) {
                        cblas_cswap(k - p - 1, &A[p + 1 + k * lda], 1, &A[p + (p + 1) * lda], lda);
                    }
                    t = A[k + k * lda];
                    A[k + k * lda] = A[p + p * lda];
                    A[p + p * lda] = t;
                }

                kk = k - kstep + 1;
                if (kp != kk) {

                    if (kp > 0) {
                        cblas_cswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    if ((kk > 0) && (kp < kk - 1)) {
                        cblas_cswap(kk - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    }
                    t = A[kk + kk * lda];
                    A[kk + kk * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = t;
                    if (kstep == 2) {
                        t = A[k - 1 + k * lda];
                        A[k - 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                }

                if (kstep == 1) {

                    if (k > 0) {

                        if (cabs1f(A[k + k * lda]) >= sfmin) {

                            d11 = CONE / A[k + k * lda];
                            csyr(uplo, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_cscal(k, &d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = A[k + k * lda];
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            csyr(uplo, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }
                    }

                } else {

                    if (k > 1) {

                        d12 = A[k - 1 + k * lda];
                        d22 = A[k - 1 + (k - 1) * lda] / d12;
                        d11 = A[k + k * lda] / d12;
                        t = CONE / (d11 * d22 - CONE);

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

            absakk = cabs1f(A[k + k * lda]);

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

                        if (!(cabs1f(A[imax + imax * lda]) < alpha * rowmax)) {

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
                    if (p > k + 1) {
                        cblas_cswap(p - k - 1, &A[k + 1 + k * lda], 1, &A[p + (k + 1) * lda], lda);
                    }
                    t = A[k + k * lda];
                    A[k + k * lda] = A[p + p * lda];
                    A[p + p * lda] = t;
                }

                kk = k + kstep - 1;
                if (kp != kk) {

                    if (kp < n - 1) {
                        cblas_cswap(n - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    if ((kk < n - 1) && (kp > kk + 1)) {
                        cblas_cswap(kp - kk - 1, &A[kk + 1 + kk * lda], 1, &A[kp + (kk + 1) * lda], lda);
                    }
                    t = A[kk + kk * lda];
                    A[kk + kk * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = t;
                    if (kstep == 2) {
                        t = A[k + 1 + k * lda];
                        A[k + 1 + k * lda] = A[kp + k * lda];
                        A[kp + k * lda] = t;
                    }
                }

                if (kstep == 1) {

                    if (k < n - 1) {

                        if (cabs1f(A[k + k * lda]) >= sfmin) {

                            d11 = CONE / A[k + k * lda];
                            csyr(uplo, n - k - 1, -d11, &A[k + 1 + k * lda], 1,
                                       &A[k + 1 + (k + 1) * lda], lda);

                            cblas_cscal(n - k - 1, &d11, &A[k + 1 + k * lda], 1);

                        } else {

                            d11 = A[k + k * lda];
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            csyr(uplo, n - k - 1, -d11, &A[k + 1 + k * lda], 1,
                                       &A[k + 1 + (k + 1) * lda], lda);
                        }
                    }

                } else {

                    if (k < n - 2) {

                        d21 = A[k + 1 + k * lda];
                        d11 = A[k + 1 + (k + 1) * lda] / d21;
                        d22 = A[k + k * lda] / d21;
                        t = CONE / (d11 * d22 - CONE);

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
