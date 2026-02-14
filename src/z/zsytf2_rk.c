/**
 * @file zsytf2_rk.c
 * @brief ZSYTF2_RK computes the factorization of a complex symmetric indefinite matrix using the bounded Bunch-Kaufman (rook) diagonal pivoting method (BLAS2 unblocked algorithm).
 */

#include "semicolon_lapack_complex_double.h"
#include <cblas.h>
#include <complex.h>
#include <math.h>

/**
 * ZSYTF2_RK computes the factorization of a complex symmetric matrix A
 * using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
 *
 *    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of the
 *          symmetric matrix A is stored:
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, contains:
 *            a) ONLY diagonal elements of the symmetric block diagonal
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
 *          elements of the symmetric block diagonal matrix D.
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
void zsytf2_rk(
    const char* uplo,
    const int n,
    c128* const restrict A,
    const int lda,
    c128* restrict E,
    int* restrict ipiv,
    int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    int upper, done;
    int i, imax = 0, j, jmax = 0, itemp, k, kk, kp, kstep, p, ii;
    f64 absakk, alpha, colmax, rowmax, dtemp, sfmin;
    c128 d11, d12, d21, d22, t, wk, wkm1, wkp1;

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
        xerbla("ZSYTF2_RK", -(*info));
        return;
    }

    alpha = (1.0 + sqrt(17.0)) / 8.0;

    sfmin = dlamch("S");

    if (upper) {

        E[0] = CZERO;

        k = n - 1;

        while (k >= 0) {

            kstep = 1;
            p = k;

            absakk = cabs1(A[k + k * lda]);

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

                        if (!(cabs1(A[imax + imax * lda]) < alpha * rowmax)) {

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
                    if (p < k - 1) {
                        cblas_zswap(k - p - 1, &A[p + 1 + k * lda], 1, &A[p + (p + 1) * lda], lda);
                    }
                    t = A[k + k * lda];
                    A[k + k * lda] = A[p + p * lda];
                    A[p + p * lda] = t;

                    if (k < n - 1) {
                        cblas_zswap(n - k - 1, &A[k + (k + 1) * lda], lda, &A[p + (k + 1) * lda], lda);
                    }
                }

                kk = k - kstep + 1;
                if (kp != kk) {

                    if (kp > 0) {
                        cblas_zswap(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }
                    if ((kk > 0) && (kp < kk - 1)) {
                        cblas_zswap(kk - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + (kp + 1) * lda], lda);
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
                        cblas_zswap(n - k - 1, &A[kk + (k + 1) * lda], lda, &A[kp + (k + 1) * lda], lda);
                    }
                }

                if (kstep == 1) {

                    if (k > 0) {

                        if (cabs1(A[k + k * lda]) >= sfmin) {

                            d11 = CONE / A[k + k * lda];
                            zsyr(uplo, k, -d11, &A[0 + k * lda], 1, A, lda);

                            cblas_zscal(k, &d11, &A[0 + k * lda], 1);

                        } else {

                            d11 = A[k + k * lda];
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            zsyr(uplo, k, -d11, &A[0 + k * lda], 1, A, lda);
                        }

                        E[k] = CZERO;
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

            absakk = cabs1(A[k + k * lda]);

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

                        if (!(cabs1(A[imax + imax * lda]) < alpha * rowmax)) {

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
                    if (p > k + 1) {
                        cblas_zswap(p - k - 1, &A[k + 1 + k * lda], 1, &A[p + (k + 1) * lda], lda);
                    }
                    t = A[k + k * lda];
                    A[k + k * lda] = A[p + p * lda];
                    A[p + p * lda] = t;

                    if (k > 0) {
                        cblas_zswap(k, &A[k + 0 * lda], lda, &A[p + 0 * lda], lda);
                    }
                }

                kk = k + kstep - 1;
                if (kp != kk) {

                    if (kp < n - 1) {
                        cblas_zswap(n - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    if ((kk < n - 1) && (kp > kk + 1)) {
                        cblas_zswap(kp - kk - 1, &A[kk + 1 + kk * lda], 1, &A[kp + (kk + 1) * lda], lda);
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
                        cblas_zswap(k, &A[kk + 0 * lda], lda, &A[kp + 0 * lda], lda);
                    }
                }

                if (kstep == 1) {

                    if (k < n - 1) {

                        if (cabs1(A[k + k * lda]) >= sfmin) {

                            d11 = CONE / A[k + k * lda];
                            zsyr(uplo, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);

                            cblas_zscal(n - k - 1, &d11, &A[k + 1 + k * lda], 1);

                        } else {

                            d11 = A[k + k * lda];
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / d11;
                            }

                            zsyr(uplo, n - k - 1, -d11,
                                       &A[k + 1 + k * lda], 1, &A[k + 1 + (k + 1) * lda], lda);
                        }

                        E[k] = CZERO;
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
