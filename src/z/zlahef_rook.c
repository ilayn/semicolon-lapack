/**
 * @file zlahef_rook.c
 * @brief ZLAHEF_ROOK computes a partial factorization of a complex Hermitian indefinite matrix using the bounded Bunch-Kaufman ("rook") diagonal pivoting method.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAHEF_ROOK computes a partial factorization of a complex Hermitian
 * matrix A using the bounded Bunch-Kaufman ("rook") diagonal pivoting
 * method. The partial factorization has the form:
 *
 * A  =  ( I  U12 ) ( A11  0  ) (  I      0     )  if UPLO = 'U', or:
 *       ( 0  U22 ) (  0   D  ) ( U12**H U22**H )
 *
 * A  =  ( L11  0 ) (  D   0  ) ( L11**H L21**H )  if UPLO = 'L'
 *       ( L21  I ) (  0  A22 ) (  0      I     )
 *
 * where the order of D is at most NB. The actual order is returned in
 * the argument KB, and is either NB or NB-1, or N if N <= NB.
 * Note that U**H denotes the conjugate transpose of U.
 *
 * ZLAHEF_ROOK is an auxiliary routine called by ZHETRF_ROOK. It uses
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
 *          On exit, details of the partial factorization.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
 *
 * @param[out] W
 *          Double complex array, dimension (ldw, nb).
 *
 * @param[in] ldw
 *          The leading dimension of the array W. ldw >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - > 0: if info = k, D(k,k) is exactly zero.
 */
void zlahef_rook(
    const char* uplo,
    const INT n,
    const INT nb,
    INT* kb,
    c128* restrict A,
    const INT lda,
    INT* restrict ipiv,
    c128* restrict W,
    const INT ldw,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 EIGHT = 8.0;
    const f64 SEVTEN = 17.0;

    INT done;
    INT imax = 0, itemp, ii, j, jb, jj, jmax = 0, jp1, jp2, k, kk, kkw, kp, kstep, kw, p;
    f64 absakk, alpha, colmax, dtemp, r1, rowmax, t, sfmin;
    c128 d11, d21, d22;

    *info = 0;

    alpha = (ONE + sqrt(SEVTEN)) / EIGHT;

    sfmin = dlamch("S");

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        k = n - 1;

        while (1) {

            kw = nb + k - (n - 1);

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

                kkw = nb + kk - (n - 1);

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
                    A[k - 1 + k * lda] = W[k - 1 + kw * ldw];
                    A[k + k * lda] = W[k + kw * ldw];

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

        /*
         * Update the upper triangle of A11 (= A(1:k,1:k)) as
         *
         * A11 := A11 - U12*D*U12**H = A11 - U12*W**H
         *
         * computing blocks of NB columns at a time (note that conjg(W) is
         * actually stored)
         *
         * Fortran: DO 50 J = ((K-1)/NB)*NB + 1, 1, -NB
         * 0-based: j goes from (k/nb)*nb down to 0 in steps of nb
         */
        for (j = (k / nb) * nb; j >= 0; j -= nb) {
            jb = nb < (k + 1 - j) ? nb : (k + 1 - j);

            for (jj = j; jj < j + jb; jj++) {
                A[jj + jj * lda] = creal(A[jj + jj * lda]);
                cblas_zgemv(CblasColMajor, CblasNoTrans, jj - j + 1, n - k - 1, &NEG_CONE,
                            &A[j + (k + 1) * lda], lda, &W[jj + (kw + 1) * ldw], ldw,
                            &CONE, &A[j + jj * lda], 1);
                A[jj + jj * lda] = creal(A[jj + jj * lda]);
            }

            if (j > 0) {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, j, jb, n - k - 1,
                            &NEG_CONE, &A[0 + (k + 1) * lda], lda, &W[j + (kw + 1) * ldw], ldw,
                            &CONE, &A[0 + j * lda], lda);
            }
        }

        /*
         * Put U12 in standard form by partially undoing the interchanges
         * in of rows in columns k+1:n looping backwards from k+1 to n
         */
        j = k + 1;
        while (j < n) {

            kstep = 1;
            jp1 = 0;
            jj = j;
            jp2 = ipiv[j];
            if (jp2 < 0) {
                jp2 = -(jp2 + 1);
                j = j + 1;
                jp1 = -(ipiv[j] + 1);
                kstep = 2;
            }

            j = j + 1;
            if (jp2 != jj && j < n) {
                cblas_zswap(n - j, &A[jp2 + j * lda], lda, &A[jj + j * lda], lda);
            }
            jj = jj + 1;
            if (kstep == 2 && jp1 != jj && j < n) {
                cblas_zswap(n - j, &A[jp1 + j * lda], lda, &A[jj + j * lda], lda);
            }
        }

        *kb = n - 1 - k;

    } else {

        k = 0;

        while (1) {

            if ((k >= nb && nb < n) || k > n - 1) {
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
                    A[k + 1 + k * lda] = W[k + 1 + k * ldw];
                    A[k + 1 + (k + 1) * lda] = W[k + 1 + (k + 1) * ldw];

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

        /*
         * Update the lower triangle of A22 (= A(k:n,k:n)) as
         *
         * A22 := A22 - L21*D*L21**H = A22 - L21*W**H
         *
         * computing blocks of NB columns at a time (note that conjg(W) is
         * actually stored)
         *
         * Fortran: DO 110 J = K, N, NB
         * 0-based: Fortran K is our k+1, Fortran N is n.
         *          j goes from k to n-1 in steps of nb (0-based)
         */
        for (j = k; j < n; j += nb) {
            jb = nb < (n - j) ? nb : (n - j);

            for (jj = j; jj < j + jb; jj++) {
                A[jj + jj * lda] = creal(A[jj + jj * lda]);
                cblas_zgemv(CblasColMajor, CblasNoTrans, j + jb - jj, k, &NEG_CONE,
                            &A[jj + 0 * lda], lda, &W[jj + 0 * ldw], ldw,
                            &CONE, &A[jj + jj * lda], 1);
                A[jj + jj * lda] = creal(A[jj + jj * lda]);
            }

            if (j + jb < n) {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, n - j - jb, jb, k,
                            &NEG_CONE, &A[j + jb + 0 * lda], lda, &W[j + 0 * ldw], ldw,
                            &CONE, &A[j + jb + j * lda], lda);
            }
        }

        /*
         * Put L21 in standard form by partially undoing the interchanges
         * of rows in columns 1:k-1 looping backwards from k-1 to 1
         */
        j = k - 1;
        while (j >= 0) {

            kstep = 1;
            jp1 = 0;
            jj = j;
            jp2 = ipiv[j];
            if (jp2 < 0) {
                jp2 = -(jp2 + 1);
                j = j - 1;
                jp1 = -(ipiv[j] + 1);
                kstep = 2;
            }

            j = j - 1;
            if (jp2 != jj && j >= 0) {
                cblas_zswap(j + 1, &A[jp2 + 0 * lda], lda, &A[jj + 0 * lda], lda);
            }
            jj = jj - 1;
            if (kstep == 2 && jp1 != jj && j >= 0) {
                cblas_zswap(j + 1, &A[jp1 + 0 * lda], lda, &A[jj + 0 * lda], lda);
            }
        }

        *kb = k;

    }
}
