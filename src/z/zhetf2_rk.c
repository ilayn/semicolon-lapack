/**
 * @file zhetf2_rk.c
 * @brief ZHETF2_RK computes the factorization of a complex Hermitian indefinite matrix using the bounded Bunch-Kaufman (rook) diagonal pivoting method (BLAS2 unblocked algorithm).
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHETF2_RK computes the factorization of a complex Hermitian matrix A
 * using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
 *
 *    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**H (or L**H) is the conjugate of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is Hermitian and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of the
 *          Hermitian matrix A is stored:
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the Hermitian matrix A.
 *          On exit, contains:
 *            a) ONLY diagonal elements of the Hermitian block diagonal
 *               matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
 *               (superdiagonal (or subdiagonal) elements of D
 *                are stored on exit in array E), and
 *            b) If UPLO = 'U': factor U in the superdiagonal part of A.
 *               If UPLO = 'L': factor L in the subdiagonal part of A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] E
 *          Double complex array, dimension (n).
 *          On exit, contains the superdiagonal (or subdiagonal)
 *          elements of the Hermitian block diagonal matrix D.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          IPIV describes the permutation matrix P in the factorization.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: If info = -k, the k-th argument had an illegal value
 *                         - > 0: If info = k, the matrix A is singular.
 */
void zhetf2_rk(
    const char* uplo,
    const int n,
    c128* restrict A,
    const int lda,
    c128* restrict E,
    int* restrict ipiv,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 EIGHT = 8.0;
    const f64 SEVTEN = 17.0;
    const c128 CZERO = CMPLX(0.0, 0.0);

    int upper, done;
    int i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f64 absakk, alpha, colmax, d11, d22, rowmax, dtemp, tt, d, r1, sfmin;
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
        xerbla("ZHETF2_RK", -(*info));
        return;
    }

    alpha = (ONE + sqrt(SEVTEN)) / EIGHT;

    sfmin = dlamch("S");

    if (upper) {

        E[0] = CZERO;

        k = n - 1;

        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = fabs(creal(A[k + k * lda]));

            if (k > 0) {
                imax = cblas_izamax(k, &A[0 + k * lda], 1);
                colmax = cabs1(A[imax + k * lda]);
            } else {
                colmax = ZERO;
            }

            if (fmax(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(A[k + k * lda]);

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
                            jmax = imax + 1 + cblas_izamax(k - imax, &A[imax + (imax + 1) * lda], lda);
                            rowmax = cabs1(A[imax + jmax * lda]);
                        } else {
                            rowmax = ZERO;
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

                    if (k < n - 1) {
                        cblas_zswap(n - k - 1, &A[k + (k + 1) * lda], lda, &A[p + (k + 1) * lda], lda);
                    }
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

                    if (k < n - 1) {
                        cblas_zswap(n - k - 1, &A[kk + (k + 1) * lda], lda, &A[kp + (k + 1) * lda], lda);
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

                            d11 = ONE / creal(A[k + k * lda]);
                            cblas_zher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_zdscal(k, d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = creal(A[k + k * lda]);
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            cblas_zher(CblasColMajor, CblasUpper, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }

                        E[k] = CZERO;
                    }

                } else {

                    if (k > 1) {

                        d = dlapy2(creal(A[k - 1 + k * lda]), cimag(A[k - 1 + k * lda]));
                        d11 = creal(A[k + k * lda]) / d;
                        d22 = creal(A[k - 1 + (k - 1) * lda]) / d;
                        d12 = A[k - 1 + k * lda] / d;
                        tt = ONE / (d11 * d22 - ONE);

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

            absakk = fabs(creal(A[k + k * lda]));

            if (k < n - 1) {
                imax = k + 1 + cblas_izamax(n - k - 1, &A[k + 1 + k * lda], 1);
                colmax = cabs1(A[imax + k * lda]);
            } else {
                colmax = ZERO;
            }

            if (fmax(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(A[k + k * lda]);

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
                            jmax = k + cblas_izamax(imax - k, &A[imax + k * lda], lda);
                            rowmax = cabs1(A[imax + jmax * lda]);
                        } else {
                            rowmax = ZERO;
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

                    if (k > 0) {
                        cblas_zswap(k, &A[k + 0 * lda], lda, &A[p + 0 * lda], lda);
                    }
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

                    if (k > 0) {
                        cblas_zswap(k, &A[kk + 0 * lda], lda, &A[kp + 0 * lda], lda);
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

                            d11 = ONE / creal(A[k + k * lda]);
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

                        E[k] = CZERO;
                    }

                } else {

                    if (k < n - 2) {

                        d = dlapy2(creal(A[k + 1 + k * lda]), cimag(A[k + 1 + k * lda]));
                        d11 = creal(A[k + 1 + (k + 1) * lda]) / d;
                        d22 = creal(A[k + k * lda]) / d;
                        d21 = A[k + 1 + k * lda] / d;
                        tt = ONE / (d11 * d22 - ONE);

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
