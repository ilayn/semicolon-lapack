/**
 * @file zlahef_rk.c
 * @brief ZLAHEF_RK computes a partial factorization of a complex Hermitian indefinite matrix using bounded Bunch-Kaufman (rook) diagonal pivoting method.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAHEF_RK computes a partial factorization of a complex Hermitian
 * matrix A using the bounded Bunch-Kaufman (rook) diagonal
 * pivoting method. The partial factorization has the form:
 *
 * A  =  ( I  U12 ) ( A11  0  ) (  I       0    )  if UPLO = 'U', or:
 *       ( 0  U22 ) (  0   D  ) ( U12**H U22**H )
 *
 * A  =  ( L11  0 ) (  D   0  ) ( L11**H L21**H )  if UPLO = 'L',
 *       ( L21  I ) (  0  A22 ) (  0       I    )
 *
 * where the order of D is at most NB. The actual order is returned in
 * the argument KB, and is either NB or NB-1, or N if N <= NB.
 *
 * ZLAHEF_RK is an auxiliary routine called by ZHETRF_RK. It uses
 * blocked code (calling Level 3 BLAS) to update the submatrix
 * A11 (if UPLO = 'U') or A22 (if UPLO = 'L').
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
 * @param[in] nb
 *          The maximum number of columns of the matrix A that should be
 *          factored. nb should be at least 2 to allow for 2-by-2 pivot
 *          blocks.
 *
 * @param[out] kb
 *          The number of columns of A that were actually factored.
 *          kb is either nb-1 or nb, or n if n <= nb.
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
 *          elements of the Hermitian block diagonal matrix D
 *          with 1-by-1 or 2-by-2 diagonal blocks.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          IPIV describes the permutation matrix P in the factorization.
 *
 * @param[out] W
 *          Double complex array, dimension (ldw, nb).
 *
 * @param[in] ldw
 *          The leading dimension of the array W. ldw >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: If info = -k, the k-th argument had an illegal value
 *                         - > 0: If info = k, the matrix A is singular.
 */
void zlahef_rk(
    const char* uplo,
    const INT n,
    const INT nb,
    INT* kb,
    c128* restrict A,
    const INT lda,
    c128* restrict E,
    INT* restrict ipiv,
    c128* restrict W,
    const INT ldw,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 EIGHT = 8.0;
    const f64 SEVTEN = 17.0;

    INT done;
    INT imax = 0, itemp, j, jmax = 0, k, kk, kw, kkw, kp, kstep, p, ii;
    f64 absakk, alpha, colmax, dtemp, r1, rowmax, sfmin, t;
    c128 d11, d21, d22;

    *info = 0;

    alpha = (ONE + sqrt(SEVTEN)) / EIGHT;

    sfmin = dlamch("S");

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        E[0] = CZERO;

        k = n - 1;

        while (1) {

            kw = nb + k - (n - 1) - 1;

            if ((k <= n - nb && nb < n) || k < 0) {
                break;
            }

            kstep = 1;
            p = k;

            if (k > 0) {
                cblas_zcopy(k, &A[0 + k * lda], 1, &W[0 + kw * ldw], 1);
            }
            W[k + kw * ldw] = creal(A[k + k * lda]);
            if (k < n - 1) {
                cblas_zgemv(CblasColMajor, CblasNoTrans, k + 1, n - k - 1, &NEG_CONE,
                            &A[0 + (k + 1) * lda], lda, &W[k + (kw + 1) * ldw], ldw,
                            &CONE, &W[0 + kw * ldw], 1);
                W[k + kw * ldw] = creal(W[k + kw * ldw]);
            }

            absakk = fabs(creal(W[k + kw * ldw]));

            if (k > 0) {
                imax = cblas_izamax(k, &W[0 + kw * ldw], 1);
                colmax = cabs1(W[imax + kw * ldw]);
            } else {
                colmax = ZERO;
            }

            if (fmax(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(W[k + kw * ldw]);
                if (k > 0) {
                    cblas_zcopy(k, &W[0 + kw * ldw], 1, &A[0 + k * lda], 1);
                }

                if (k > 0) {
                    E[k] = CZERO;
                }

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        if (imax > 0) {
                            cblas_zcopy(imax, &A[0 + imax * lda], 1, &W[0 + (kw - 1) * ldw], 1);
                        }
                        W[imax + (kw - 1) * ldw] = creal(A[imax + imax * lda]);

                        cblas_zcopy(k - imax, &A[imax + (imax + 1) * lda], lda,
                                    &W[imax + 1 + (kw - 1) * ldw], 1);
                        zlacgv(k - imax, &W[imax + 1 + (kw - 1) * ldw], 1);

                        if (k < n - 1) {
                            cblas_zgemv(CblasColMajor, CblasNoTrans, k + 1, n - k - 1, &NEG_CONE,
                                        &A[0 + (k + 1) * lda], lda, &W[imax + (kw + 1) * ldw], ldw,
                                        &CONE, &W[0 + (kw - 1) * ldw], 1);
                            W[imax + (kw - 1) * ldw] = creal(W[imax + (kw - 1) * ldw]);
                        }

                        if (imax != k) {
                            jmax = imax + 1 + cblas_izamax(k - imax, &W[imax + 1 + (kw - 1) * ldw], 1);
                            rowmax = cabs1(W[jmax + (kw - 1) * ldw]);
                        } else {
                            rowmax = ZERO;
                        }

                        if (imax > 0) {
                            itemp = cblas_izamax(imax, &W[0 + (kw - 1) * ldw], 1);
                            dtemp = cabs1(W[itemp + (kw - 1) * ldw]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabs(creal(W[imax + (kw - 1) * ldw])) < alpha * rowmax)) {

                            kp = imax;

                            cblas_zcopy(k + 1, &W[0 + (kw - 1) * ldw], 1, &W[0 + kw * ldw], 1);

                            done = 1;

                        } else if ((p == jmax) || (rowmax <= colmax)) {

                            kp = imax;
                            kstep = 2;
                            done = 1;

                        } else {

                            p = imax;
                            colmax = rowmax;
                            imax = jmax;

                            cblas_zcopy(k + 1, &W[0 + (kw - 1) * ldw], 1, &W[0 + kw * ldw], 1);

                        }
                    }
                }

                kk = k - kstep + 1;

                kkw = nb + kk - (n - 1) - 1;

                if ((kstep == 2) && (p != k)) {

                    A[p + p * lda] = creal(A[k + k * lda]);
                    cblas_zcopy(k - 1 - p, &A[p + 1 + k * lda], 1, &A[p + (p + 1) * lda], lda);
                    zlacgv(k - 1 - p, &A[p + (p + 1) * lda], lda);
                    if (p > 0) {
                        cblas_zcopy(p, &A[0 + k * lda], 1, &A[0 + p * lda], 1);
                    }

                    if (k < n - 1) {
                        cblas_zswap(n - k - 1, &A[k + (k + 1) * lda], lda, &A[p + (k + 1) * lda], lda);
                    }
                    cblas_zswap(n - kk, &W[k + kkw * ldw], ldw, &W[p + kkw * ldw], ldw);
                }

                if (kp != kk) {

                    A[kp + kp * lda] = creal(A[kk + kk * lda]);
                    cblas_zcopy(kk - 1 - kp, &A[kp + 1 + kk * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    zlacgv(kk - 1 - kp, &A[kp + (kp + 1) * lda], lda);
                    if (kp > 0) {
                        cblas_zcopy(kp, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);
                    }

                    if (k < n - 1) {
                        cblas_zswap(n - k - 1, &A[kk + (k + 1) * lda], lda, &A[kp + (k + 1) * lda], lda);
                    }
                    cblas_zswap(n - kk, &W[kk + kkw * ldw], ldw, &W[kp + kkw * ldw], ldw);
                }

                if (kstep == 1) {

                    cblas_zcopy(k + 1, &W[0 + kw * ldw], 1, &A[0 + k * lda], 1);
                    if (k > 0) {

                        t = creal(A[k + k * lda]);
                        if (fabs(t) >= sfmin) {
                            r1 = ONE / t;
                            cblas_zdscal(k, r1, &A[0 + k * lda], 1);
                        } else if (t != ZERO) {
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / t;
                            }
                        }

                        zlacgv(k, &W[0 + kw * ldw], 1);

                        E[k] = CZERO;
                    }

                } else {

                    if (k > 1) {

                        d21 = W[k - 1 + kw * ldw];
                        d11 = W[k + kw * ldw] / conj(d21);
                        d22 = W[k - 1 + (kw - 1) * ldw] / d21;
                        t = ONE / (creal(d11 * d22) - ONE);
                        for (j = 0; j <= k - 2; j++) {
                            A[j + (k - 1) * lda] = t * ((d11 * W[j + (kw - 1) * ldw] - W[j + kw * ldw]) / d21);
                            A[j + k * lda] = t * ((d22 * W[j + kw * ldw] - W[j + (kw - 1) * ldw]) / conj(d21));
                        }
                    }

                    A[k - 1 + (k - 1) * lda] = W[k - 1 + (kw - 1) * ldw];
                    A[k - 1 + k * lda] = CZERO;
                    A[k + k * lda] = W[k + kw * ldw];
                    E[k] = W[k - 1 + kw * ldw];
                    E[k - 1] = CZERO;

                    zlacgv(k, &W[0 + kw * ldw], 1);
                    zlacgv(k - 1, &W[0 + (kw - 1) * ldw], 1);

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

        if (k >= 0) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        k + 1, k + 1, n - k - 1, &NEG_CONE,
                        &A[0 + (k + 1) * lda], lda, &W[0 + (kw + 1) * ldw], ldw,
                        &CONE, &A[0], lda);
        }

        *kb = n - k - 1;

    } else {

        E[n - 1] = CZERO;

        k = 0;

        while (1) {

            if ((k >= nb - 1 && nb < n) || k > n - 1) {
                break;
            }

            kstep = 1;
            p = k;

            W[k + k * ldw] = creal(A[k + k * lda]);
            if (k < n - 1) {
                cblas_zcopy(n - k - 1, &A[k + 1 + k * lda], 1, &W[k + 1 + k * ldw], 1);
            }
            if (k > 0) {
                cblas_zgemv(CblasColMajor, CblasNoTrans, n - k, k, &NEG_CONE,
                            &A[k + 0 * lda], lda, &W[k + 0 * ldw], ldw,
                            &CONE, &W[k + k * ldw], 1);
                W[k + k * ldw] = creal(W[k + k * ldw]);
            }

            absakk = fabs(creal(W[k + k * ldw]));

            if (k < n - 1) {
                imax = k + 1 + cblas_izamax(n - k - 1, &W[k + 1 + k * ldw], 1);
                colmax = cabs1(W[imax + k * ldw]);
            } else {
                colmax = ZERO;
            }

            if (fmax(absakk, colmax) == ZERO) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(W[k + k * ldw]);
                if (k < n - 1) {
                    cblas_zcopy(n - k - 1, &W[k + 1 + k * ldw], 1, &A[k + 1 + k * lda], 1);
                }

                if (k < n - 1) {
                    E[k] = CZERO;
                }

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        cblas_zcopy(imax - k, &A[imax + k * lda], lda, &W[k + (k + 1) * ldw], 1);
                        zlacgv(imax - k, &W[k + (k + 1) * ldw], 1);
                        W[imax + (k + 1) * ldw] = creal(A[imax + imax * lda]);

                        if (imax < n - 1) {
                            cblas_zcopy(n - imax - 1, &A[imax + 1 + imax * lda], 1,
                                        &W[imax + 1 + (k + 1) * ldw], 1);
                        }

                        if (k > 0) {
                            cblas_zgemv(CblasColMajor, CblasNoTrans, n - k, k, &NEG_CONE,
                                        &A[k + 0 * lda], lda, &W[imax + 0 * ldw], ldw,
                                        &CONE, &W[k + (k + 1) * ldw], 1);
                            W[imax + (k + 1) * ldw] = creal(W[imax + (k + 1) * ldw]);
                        }

                        if (imax != k) {
                            jmax = k + cblas_izamax(imax - k, &W[k + (k + 1) * ldw], 1);
                            rowmax = cabs1(W[jmax + (k + 1) * ldw]);
                        } else {
                            rowmax = ZERO;
                        }

                        if (imax < n - 1) {
                            itemp = imax + 1 + cblas_izamax(n - imax - 1, &W[imax + 1 + (k + 1) * ldw], 1);
                            dtemp = cabs1(W[itemp + (k + 1) * ldw]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(fabs(creal(W[imax + (k + 1) * ldw])) < alpha * rowmax)) {

                            kp = imax;

                            cblas_zcopy(n - k, &W[k + (k + 1) * ldw], 1, &W[k + k * ldw], 1);

                            done = 1;

                        } else if ((p == jmax) || (rowmax <= colmax)) {

                            kp = imax;
                            kstep = 2;
                            done = 1;

                        } else {

                            p = imax;
                            colmax = rowmax;
                            imax = jmax;

                            cblas_zcopy(n - k, &W[k + (k + 1) * ldw], 1, &W[k + k * ldw], 1);

                        }
                    }
                }

                kk = k + kstep - 1;

                if ((kstep == 2) && (p != k)) {

                    A[p + p * lda] = creal(A[k + k * lda]);
                    cblas_zcopy(p - k - 1, &A[k + 1 + k * lda], 1, &A[p + (k + 1) * lda], lda);
                    zlacgv(p - k - 1, &A[p + (k + 1) * lda], lda);
                    if (p < n - 1) {
                        cblas_zcopy(n - p - 1, &A[p + 1 + k * lda], 1, &A[p + 1 + p * lda], 1);
                    }

                    if (k > 0) {
                        cblas_zswap(k, &A[k + 0 * lda], lda, &A[p + 0 * lda], lda);
                    }
                    cblas_zswap(kk + 1, &W[k + 0 * ldw], ldw, &W[p + 0 * ldw], ldw);
                }

                if (kp != kk) {

                    A[kp + kp * lda] = creal(A[kk + kk * lda]);
                    cblas_zcopy(kp - kk - 1, &A[kk + 1 + kk * lda], 1, &A[kp + (kk + 1) * lda], lda);
                    zlacgv(kp - kk - 1, &A[kp + (kk + 1) * lda], lda);
                    if (kp < n - 1) {
                        cblas_zcopy(n - kp - 1, &A[kp + 1 + kk * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }

                    if (k > 0) {
                        cblas_zswap(k, &A[kk + 0 * lda], lda, &A[kp + 0 * lda], lda);
                    }
                    cblas_zswap(kk + 1, &W[kk + 0 * ldw], ldw, &W[kp + 0 * ldw], ldw);
                }

                if (kstep == 1) {

                    cblas_zcopy(n - k, &W[k + k * ldw], 1, &A[k + k * lda], 1);
                    if (k < n - 1) {

                        t = creal(A[k + k * lda]);
                        if (fabs(t) >= sfmin) {
                            r1 = ONE / t;
                            cblas_zdscal(n - k - 1, r1, &A[k + 1 + k * lda], 1);
                        } else if (t != ZERO) {
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / t;
                            }
                        }

                        zlacgv(n - k - 1, &W[k + 1 + k * ldw], 1);

                        E[k] = CZERO;
                    }

                } else {

                    if (k < n - 2) {

                        d21 = W[k + 1 + k * ldw];
                        d11 = W[k + 1 + (k + 1) * ldw] / d21;
                        d22 = W[k + k * ldw] / conj(d21);
                        t = ONE / (creal(d11 * d22) - ONE);
                        for (j = k + 2; j < n; j++) {
                            A[j + k * lda] = t * ((d11 * W[j + k * ldw] - W[j + (k + 1) * ldw]) / conj(d21));
                            A[j + (k + 1) * lda] = t * ((d22 * W[j + (k + 1) * ldw] - W[j + k * ldw]) / d21);
                        }
                    }

                    A[k + k * lda] = W[k + k * ldw];
                    A[k + 1 + k * lda] = CZERO;
                    A[k + 1 + (k + 1) * lda] = W[k + 1 + (k + 1) * ldw];
                    E[k] = W[k + 1 + k * ldw];
                    E[k + 1] = CZERO;

                    zlacgv(n - k - 1, &W[k + 1 + k * ldw], 1);
                    zlacgv(n - k - 2, &W[k + 2 + (k + 1) * ldw], 1);

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

        if (k < n) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        n - k, n - k, k, &NEG_CONE,
                        &A[k + 0 * lda], lda, &W[k + 0 * ldw], ldw,
                        &CONE, &A[k + k * lda], lda);
        }

        *kb = k;

    }
}
