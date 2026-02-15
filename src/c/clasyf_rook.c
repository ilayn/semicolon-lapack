/**
 * @file clasyf_rook.c
 * @brief CLASYF_ROOK computes a partial factorization of a complex symmetric matrix using the bounded Bunch-Kaufman ("rook") diagonal pivoting method.
 */

#include "semicolon_lapack_complex_single.h"
#include <cblas.h>
#include <complex.h>
#include <math.h>

/**
 * CLASYF_ROOK computes a partial factorization of a complex symmetric
 * matrix A using the bounded Bunch-Kaufman ("rook") diagonal
 * pivoting method.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nb
 *          The maximum number of columns to factor. nb >= 2.
 *
 * @param[out] kb
 *          The number of columns actually factored.
 *
 * @param[in,out] A
 *          Single complex array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, details of the partial factorization.
 *
 * @param[in] lda
 *          The leading dimension of A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and block structure.
 *
 * @param[out] W
 *          Single complex array, dimension (ldw, nb).
 *
 * @param[in] ldw
 *          The leading dimension of W. ldw >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - > 0: if info = k, D(k,k) is exactly zero.
 */
void clasyf_rook(
    const char* uplo,
    const int n,
    const int nb,
    int* kb,
    c64* restrict A,
    const int lda,
    int* restrict ipiv,
    c64* restrict W,
    const int ldw,
    int* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

    int done;
    int imax = 0, itemp, j, jj, jmax = 0, jp1, jp2, k, kk, kw, kkw, kp, kstep, p, ii;
    f32 absakk, alpha, colmax, dtemp, rowmax, sfmin;
    c64 d11, d12, d21, d22, r1, t;

    *info = 0;

    alpha = (1.0f + sqrtf(17.0f)) / 8.0f;

    sfmin = slamch("S");

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        k = n - 1;
        while (1) {

            kw = nb + k - (n - 1);

            if ((k <= n - nb && nb < n) || k < 0) {
                break;
            }

            kstep = 1;
            p = k;

            cblas_ccopy(k + 1, &A[0 + k * lda], 1, &W[0 + kw * ldw], 1);
            if (k < n - 1) {
                cblas_cgemv(CblasColMajor, CblasNoTrans, k + 1, n - k - 1, &NEG_CONE,
                            &A[0 + (k + 1) * lda], lda, &W[k + (kw + 1) * ldw], ldw,
                            &CONE, &W[0 + kw * ldw], 1);
            }

            absakk = cabs1f(W[k + kw * ldw]);

            if (k > 0) {
                imax = cblas_icamax(k, &W[0 + kw * ldw], 1);
                colmax = cabs1f(W[imax + kw * ldw]);
            } else {
                colmax = 0.0f;
            }

            if (fmaxf(absakk, colmax) == 0.0f) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                cblas_ccopy(k + 1, &W[0 + kw * ldw], 1, &A[0 + k * lda], 1);

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        cblas_ccopy(imax + 1, &A[0 + imax * lda], 1, &W[0 + (kw - 1) * ldw], 1);
                        cblas_ccopy(k - imax, &A[imax + (imax + 1) * lda], lda,
                                    &W[imax + 1 + (kw - 1) * ldw], 1);

                        if (k < n - 1) {
                            cblas_cgemv(CblasColMajor, CblasNoTrans, k + 1, n - k - 1, &NEG_CONE,
                                        &A[0 + (k + 1) * lda], lda, &W[imax + (kw + 1) * ldw], ldw,
                                        &CONE, &W[0 + (kw - 1) * ldw], 1);
                        }

                        if (imax != k) {
                            jmax = imax + 1 + cblas_icamax(k - imax, &W[imax + 1 + (kw - 1) * ldw], 1);
                            rowmax = cabs1f(W[jmax + (kw - 1) * ldw]);
                        } else {
                            rowmax = 0.0f;
                        }

                        if (imax > 0) {
                            itemp = cblas_icamax(imax, &W[0 + (kw - 1) * ldw], 1);
                            dtemp = cabs1f(W[itemp + (kw - 1) * ldw]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(cabs1f(W[imax + (kw - 1) * ldw]) < alpha * rowmax)) {

                            kp = imax;

                            cblas_ccopy(k + 1, &W[0 + (kw - 1) * ldw], 1, &W[0 + kw * ldw], 1);

                            done = 1;

                        } else if ((p == jmax) || (rowmax <= colmax)) {

                            kp = imax;
                            kstep = 2;
                            done = 1;

                        } else {

                            p = imax;
                            colmax = rowmax;
                            imax = jmax;

                            cblas_ccopy(k + 1, &W[0 + (kw - 1) * ldw], 1, &W[0 + kw * ldw], 1);

                        }
                    }
                }

                kk = k - kstep + 1;

                kkw = nb + kk - (n - 1);

                if ((kstep == 2) && (p != k)) {

                    cblas_ccopy(k - p, &A[p + 1 + k * lda], 1, &A[p + (p + 1) * lda], lda);
                    cblas_ccopy(p + 1, &A[0 + k * lda], 1, &A[0 + p * lda], 1);

                    cblas_cswap(n - k, &A[k + k * lda], lda, &A[p + k * lda], lda);
                    cblas_cswap(n - kk, &W[k + kkw * ldw], ldw, &W[p + kkw * ldw], ldw);
                }

                if (kp != kk) {

                    A[kp + k * lda] = A[kk + k * lda];
                    cblas_ccopy(k - 1 - kp, &A[kp + 1 + kk * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    cblas_ccopy(kp + 1, &A[0 + kk * lda], 1, &A[0 + kp * lda], 1);

                    cblas_cswap(n - kk, &A[kk + kk * lda], lda, &A[kp + kk * lda], lda);
                    cblas_cswap(n - kk, &W[kk + kkw * ldw], ldw, &W[kp + kkw * ldw], ldw);
                }

                if (kstep == 1) {

                    cblas_ccopy(k + 1, &W[0 + kw * ldw], 1, &A[0 + k * lda], 1);
                    if (k > 0) {
                        if (cabs1f(A[k + k * lda]) >= sfmin) {
                            r1 = CONE / A[k + k * lda];
                            cblas_cscal(k, &r1, &A[0 + k * lda], 1);
                        } else if (A[k + k * lda] != CZERO) {
                            for (ii = 0; ii < k; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / A[k + k * lda];
                            }
                        }
                    }

                } else {

                    if (k > 1) {

                        d12 = W[k - 1 + kw * ldw];
                        d11 = W[k + kw * ldw] / d12;
                        d22 = W[k - 1 + (kw - 1) * ldw] / d12;
                        t = CONE / (d11 * d22 - CONE);
                        for (j = 0; j <= k - 2; j++) {
                            A[j + (k - 1) * lda] = t * ((d11 * W[j + (kw - 1) * ldw] - W[j + kw * ldw]) / d12);
                            A[j + k * lda] = t * ((d22 * W[j + kw * ldw] - W[j + (kw - 1) * ldw]) / d12);
                        }
                    }

                    A[k - 1 + (k - 1) * lda] = W[k - 1 + (kw - 1) * ldw];
                    A[k - 1 + k * lda] = W[k - 1 + kw * ldw];
                    A[k + k * lda] = W[k + kw * ldw];
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

        /* Update the upper triangle of A11 using cblas_zgemm (ZGEMMTR replacement) */
        if (k >= 0 && n > nb) {
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        k + 1, k + 1, n - k - 1, &NEG_CONE,
                        &A[0 + (k + 1) * lda], lda, &W[0 + (kw + 1) * ldw], ldw,
                        &CONE, &A[0], lda);
        }

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
                cblas_cswap(n - j, &A[jp2 + j * lda], lda, &A[jj + j * lda], lda);
            }
            jj = j - 1;
            if (jp1 != jj && kstep == 2 && j < n) {
                cblas_cswap(n - j, &A[jp1 + j * lda], lda, &A[jj + j * lda], lda);
            }
        }

        *kb = n - 1 - k;

    } else {

        k = 0;
        while (1) {

            if ((k >= nb - 1 && nb < n) || k >= n) {
                break;
            }

            kstep = 1;
            p = k;

            cblas_ccopy(n - k, &A[k + k * lda], 1, &W[k + k * ldw], 1);
            if (k > 0) {
                cblas_cgemv(CblasColMajor, CblasNoTrans, n - k, k, &NEG_CONE,
                            &A[k + 0 * lda], lda, &W[k + 0 * ldw], ldw,
                            &CONE, &W[k + k * ldw], 1);
            }

            absakk = cabs1f(W[k + k * ldw]);

            if (k < n - 1) {
                imax = k + 1 + cblas_icamax(n - k - 1, &W[k + 1 + k * ldw], 1);
                colmax = cabs1f(W[imax + k * ldw]);
            } else {
                colmax = 0.0f;
            }

            if (fmaxf(absakk, colmax) == 0.0f) {

                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                cblas_ccopy(n - k, &W[k + k * ldw], 1, &A[k + k * lda], 1);

            } else {

                if (!(absakk < alpha * colmax)) {

                    kp = k;

                } else {

                    done = 0;

                    while (!done) {

                        cblas_ccopy(imax - k, &A[imax + k * lda], lda, &W[k + (k + 1) * ldw], 1);
                        cblas_ccopy(n - imax, &A[imax + imax * lda], 1, &W[imax + (k + 1) * ldw], 1);
                        if (k > 0) {
                            cblas_cgemv(CblasColMajor, CblasNoTrans, n - k, k, &NEG_CONE,
                                        &A[k + 0 * lda], lda, &W[imax + 0 * ldw], ldw,
                                        &CONE, &W[k + (k + 1) * ldw], 1);
                        }

                        if (imax != k) {
                            jmax = k + cblas_icamax(imax - k, &W[k + (k + 1) * ldw], 1);
                            rowmax = cabs1f(W[jmax + (k + 1) * ldw]);
                        } else {
                            rowmax = 0.0f;
                        }

                        if (imax < n - 1) {
                            itemp = imax + 1 + cblas_icamax(n - imax - 1, &W[imax + 1 + (k + 1) * ldw], 1);
                            dtemp = cabs1f(W[itemp + (k + 1) * ldw]);
                            if (dtemp > rowmax) {
                                rowmax = dtemp;
                                jmax = itemp;
                            }
                        }

                        if (!(cabs1f(W[imax + (k + 1) * ldw]) < alpha * rowmax)) {

                            kp = imax;

                            cblas_ccopy(n - k, &W[k + (k + 1) * ldw], 1, &W[k + k * ldw], 1);

                            done = 1;

                        } else if ((p == jmax) || (rowmax <= colmax)) {

                            kp = imax;
                            kstep = 2;
                            done = 1;

                        } else {

                            p = imax;
                            colmax = rowmax;
                            imax = jmax;

                            cblas_ccopy(n - k, &W[k + (k + 1) * ldw], 1, &W[k + k * ldw], 1);

                        }
                    }
                }

                kk = k + kstep - 1;

                if ((kstep == 2) && (p != k)) {

                    cblas_ccopy(p - k, &A[k + k * lda], 1, &A[p + k * lda], lda);
                    cblas_ccopy(n - p, &A[p + k * lda], 1, &A[p + p * lda], 1);

                    cblas_cswap(k + 1, &A[k + 0 * lda], lda, &A[p + 0 * lda], lda);
                    cblas_cswap(kk + 1, &W[k + 0 * ldw], ldw, &W[p + 0 * ldw], ldw);
                }

                if (kp != kk) {

                    A[kp + k * lda] = A[kk + k * lda];
                    cblas_ccopy(kp - k - 1, &A[k + 1 + kk * lda], 1, &A[kp + (k + 1) * lda], lda);
                    cblas_ccopy(n - kp, &A[kp + kk * lda], 1, &A[kp + kp * lda], 1);

                    cblas_cswap(kk + 1, &A[kk + 0 * lda], lda, &A[kp + 0 * lda], lda);
                    cblas_cswap(kk + 1, &W[kk + 0 * ldw], ldw, &W[kp + 0 * ldw], ldw);
                }

                if (kstep == 1) {

                    cblas_ccopy(n - k, &W[k + k * ldw], 1, &A[k + k * lda], 1);
                    if (k < n - 1) {
                        if (cabs1f(A[k + k * lda]) >= sfmin) {
                            r1 = CONE / A[k + k * lda];
                            cblas_cscal(n - k - 1, &r1, &A[k + 1 + k * lda], 1);
                        } else if (A[k + k * lda] != CZERO) {
                            for (ii = k + 1; ii < n; ii++) {
                                A[ii + k * lda] = A[ii + k * lda] / A[k + k * lda];
                            }
                        }
                    }

                } else {

                    if (k < n - 2) {

                        d21 = W[k + 1 + k * ldw];
                        d11 = W[k + 1 + (k + 1) * ldw] / d21;
                        d22 = W[k + k * ldw] / d21;
                        t = CONE / (d11 * d22 - CONE);
                        for (j = k + 2; j < n; j++) {
                            A[j + k * lda] = t * ((d11 * W[j + k * ldw] - W[j + (k + 1) * ldw]) / d21);
                            A[j + (k + 1) * lda] = t * ((d22 * W[j + (k + 1) * ldw] - W[j + k * ldw]) / d21);
                        }
                    }

                    A[k + k * lda] = W[k + k * ldw];
                    A[k + 1 + k * lda] = W[k + 1 + k * ldw];
                    A[k + 1 + (k + 1) * lda] = W[k + 1 + (k + 1) * ldw];
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

        /* Update the lower triangle of A22 using cblas_zgemm (ZGEMMTR replacement) */
        if (k < n && k > 0) {
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        n - k, n - k, k, &NEG_CONE,
                        &A[k + 0 * lda], lda, &W[k + 0 * ldw], ldw,
                        &CONE, &A[k + k * lda], lda);
        }

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
                cblas_cswap(j + 1, &A[jp2 + 0 * lda], lda, &A[jj + 0 * lda], lda);
            }
            jj = j + 1;
            if (jp1 != jj && kstep == 2 && j >= 0) {
                cblas_cswap(j + 1, &A[jp1 + 0 * lda], lda, &A[jj + 0 * lda], lda);
            }
        }

        *kb = k;

    }
}
