/**
 * @file zlahef.c
 * @brief ZLAHEF computes a partial factorization of a complex Hermitian
 *        indefinite matrix using the Bunch-Kaufman diagonal pivoting method.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/* Alpha for Bunch-Kaufman pivoting: (1 + sqrt(17)) / 8 */
static const double ALPHA_BK = 0.6403882032022076;

/**
 * ZLAHEF computes a partial factorization of a complex Hermitian
 * matrix A using the Bunch-Kaufman diagonal pivoting method. The
 * partial factorization has the form:
 *
 *    A = ( I  U12 ) ( A11  0  ) (  I      0     )  if UPLO = 'U', or:
 *        ( 0  U22 ) (  0   D  ) ( U12**H U22**H )
 *
 *    A = ( L11  0 ) (  D   0  ) ( L11**H L21**H )  if UPLO = 'L'
 *        ( L21  I ) (  0  A22 ) (  0      I     )
 *
 * where the order of D is at most NB. The actual order is returned in
 * the argument KB, and is either NB or NB-1, or N if N <= NB.
 * Note that U**H denotes the conjugate transpose of U.
 *
 * ZLAHEF is an auxiliary routine called by ZHETRF. It uses blocked code
 * (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = 'U') or
 * A22 (if UPLO = 'L').
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the Hermitian matrix A is stored:
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nb    The maximum number of columns of the matrix A that
 *                      should be factored. nb should be at least 2 to allow
 *                      for 2-by-2 pivot blocks.
 * @param[out]    kb    The number of columns of A that were actually factored.
 *                      kb is either nb-1 or nb, or n if n <= nb.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A. If uplo = 'U', the
 *                      leading n-by-n upper triangular part contains the upper
 *                      triangular part. If uplo = 'L', the leading n-by-n lower
 *                      triangular part contains the lower triangular part.
 *                      On exit, A contains details of the partial factorization.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    ipiv  Integer array, dimension (n). Details of the
 *                      interchanges and the block structure of D.
 *                      If uplo = 'U': Only the last kb elements of ipiv are set.
 *                        If ipiv[k] >= 0, rows and columns k and ipiv[k] were
 *                        interchanged, D(k,k) is a 1-by-1 diagonal block.
 *                        If ipiv[k] < 0, rows and columns k-1 and -(ipiv[k]+1)
 *                        were interchanged, D(k-1:k,k-1:k) is a 2-by-2 block,
 *                        and ipiv[k-1] = ipiv[k].
 *                      If uplo = 'L': Only the first kb elements of ipiv are set.
 *                        If ipiv[k] >= 0, rows and columns k and ipiv[k] were
 *                        interchanged, D(k,k) is a 1-by-1 diagonal block.
 *                        If ipiv[k] < 0, rows and columns k+1 and -(ipiv[k]+1)
 *                        were interchanged, D(k:k+1,k:k+1) is a 2-by-2 block,
 *                        and ipiv[k+1] = ipiv[k].
 * @param[out]    W     Complex array, dimension (ldw, nb).
 *                      Workspace for storing updated columns during factorization.
 * @param[in]     ldw   The leading dimension of the array W. ldw >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - > 0: if info = k+1, D(k,k) is exactly zero. The
 *                           factorization has been completed, but the block
 *                           diagonal matrix D is exactly singular.
 */
void zlahef(
    const char* uplo,
    const int n,
    const int nb,
    int* kb,
    double complex* restrict A,
    const int lda,
    int* restrict ipiv,
    double complex* restrict W,
    const int ldw,
    int* info)
{
    const double complex CONE = CMPLX(1.0, 0.0);
    const double complex NEG_CONE = CMPLX(-1.0, 0.0);

    *info = 0;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /* Factorize the trailing columns of A using the upper triangle
         * of A and working backwards, and compute the matrix W = U12*D
         * for use in updating A11 (note that conjg(W) is actually stored)
         *
         * K is the main loop index, decreasing from N in steps of 1 or 2.
         *
         * KW is the column of W which corresponds to column K of A. */

        int k = n - 1;

        while (1) {
            int kw = nb - 1 - (n - 1 - k);

            if ((k <= n - nb && nb < n) || k < 0) {
                break;
            }

            int kstep = 1;

            /* Copy column K of A to column KW of W and update it */
            if (k > 0) {
                cblas_zcopy(k, &A[0 + k * lda], 1, &W[0 + kw * ldw], 1);
            }
            W[k + kw * ldw] = creal(A[k + k * lda]);
            if (k < n - 1) {
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            k + 1, n - 1 - k,
                            &NEG_CONE, &A[0 + (k + 1) * lda], lda,
                            &W[k + (kw + 1) * ldw], ldw,
                            &CONE, &W[0 + kw * ldw], 1);
                W[k + kw * ldw] = creal(W[k + kw * ldw]);
            }

            double absakk = fabs(creal(W[k + kw * ldw]));

            /* IMAX is the row-index of the largest off-diagonal element in
             * column K, and COLMAX is its absolute value.
             * Determine both COLMAX and IMAX. */
            int imax = 0;
            double colmax = 0.0;
            if (k > 0) {
                imax = cblas_izamax(k, &W[0 + kw * ldw], 1);
                colmax = cabs1(W[imax + kw * ldw]);
            }

            int kp;

            if (fmax(absakk, colmax) == 0.0) {
                /* Column K is zero or underflow: set INFO and continue */
                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(A[k + k * lda]);
            } else {

                if (absakk >= ALPHA_BK * colmax) {
                    /* No interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* Copy column IMAX to column KW-1 of W and update it */
                    if (imax > 0) {
                        cblas_zcopy(imax, &A[0 + imax * lda], 1,
                                    &W[0 + (kw - 1) * ldw], 1);
                    }
                    W[imax + (kw - 1) * ldw] = creal(A[imax + imax * lda]);

                    if (k - imax > 0) {
                        cblas_zcopy(k - imax, &A[imax + (imax + 1) * lda], lda,
                                    &W[(imax + 1) + (kw - 1) * ldw], 1);
                        zlacgv(k - imax, &W[(imax + 1) + (kw - 1) * ldw], 1);
                    }

                    if (k < n - 1) {
                        cblas_zgemv(CblasColMajor, CblasNoTrans,
                                    k + 1, n - 1 - k,
                                    &NEG_CONE, &A[0 + (k + 1) * lda], lda,
                                    &W[imax + (kw + 1) * ldw], ldw,
                                    &CONE, &W[0 + (kw - 1) * ldw], 1);
                        W[imax + (kw - 1) * ldw] = creal(W[imax + (kw - 1) * ldw]);
                    }

                    /* JMAX is the column-index of the largest off-diagonal
                     * element in row IMAX, and ROWMAX is its absolute value */
                    int jmax = (imax + 1) + cblas_izamax(k - imax, &W[(imax + 1) + (kw - 1) * ldw], 1);
                    double rowmax = cabs1(W[jmax + (kw - 1) * ldw]);

                    if (imax > 0) {
                        jmax = cblas_izamax(imax, &W[0 + (kw - 1) * ldw], 1);
                        rowmax = fmax(rowmax, cabs1(W[jmax + (kw - 1) * ldw]));
                    }

                    if (absakk >= ALPHA_BK * colmax * (colmax / rowmax)) {
                        /* No interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (fabs(creal(W[imax + (kw - 1) * ldw])) >= ALPHA_BK * rowmax) {
                        /* Interchange rows and columns K and IMAX,
                         * use 1-by-1 pivot block.
                         * Copy column KW-1 of W to column KW of W. */
                        kp = imax;
                        cblas_zcopy(k + 1, &W[0 + (kw - 1) * ldw], 1,
                                    &W[0 + kw * ldw], 1);
                    } else {
                        /* Interchange rows and columns K-1 and IMAX,
                         * use 2-by-2 pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                /* KK is the column of A where pivoting step stopped */
                int kk = k - kstep + 1;

                /* KKW is the column of W which corresponds to column KK of A */
                int kkw = nb - 1 - (n - 1 - kk);

                /* Interchange rows and columns KP and KK.
                 * Updated column KP is already stored in column KKW of W. */
                if (kp != kk) {
                    A[kp + kp * lda] = creal(A[kk + kk * lda]);

                    if (kk - kp - 1 > 0) {
                        cblas_zcopy(kk - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[kp + (kp + 1) * lda], lda);
                        zlacgv(kk - kp - 1, &A[kp + (kp + 1) * lda], lda);
                    }

                    if (kp > 0) {
                        cblas_zcopy(kp, &A[0 + kk * lda], 1,
                                    &A[0 + kp * lda], 1);
                    }

                    if (k < n - 1) {
                        cblas_zswap(n - 1 - k, &A[kk + (k + 1) * lda], lda,
                                    &A[kp + (k + 1) * lda], lda);
                    }

                    cblas_zswap(nb - kkw, &W[kk + kkw * ldw], ldw,
                                &W[kp + kkw * ldw], ldw);
                }

                if (kstep == 1) {

                    cblas_zcopy(k + 1, &W[0 + kw * ldw], 1, &A[0 + k * lda], 1);

                    if (k > 0) {
                        double r1 = 1.0 / creal(A[k + k * lda]);
                        cblas_zdscal(k, r1, &A[0 + k * lda], 1);

                        zlacgv(k, &W[0 + kw * ldw], 1);
                    }

                } else {

                    if (k > 1) {
                        double complex d21 = W[(k - 1) + kw * ldw];
                        double complex d11 = W[k + kw * ldw] / conj(d21);
                        double complex d22 = W[(k - 1) + (kw - 1) * ldw] / d21;
                        double t = 1.0 / (creal(d11 * d22) - 1.0);
                        d21 = t / d21;

                        for (int j = 0; j <= k - 2; j++) {
                            A[j + (k - 1) * lda] = d21 * (d11 * W[j + (kw - 1) * ldw] - W[j + kw * ldw]);
                            A[j + k * lda] = conj(d21) * (d22 * W[j + kw * ldw] - W[j + (kw - 1) * ldw]);
                        }
                    }

                    A[(k - 1) + (k - 1) * lda] = W[(k - 1) + (kw - 1) * ldw];
                    A[(k - 1) + k * lda] = W[(k - 1) + kw * ldw];
                    A[k + k * lda] = W[k + kw * ldw];

                    zlacgv(k, &W[0 + kw * ldw], 1);
                    if (k > 1) {
                        zlacgv(k - 1, &W[0 + (kw - 1) * ldw], 1);
                    }
                }
            }

            /* Store details of the interchanges in IPIV */
            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(kp + 1);
                ipiv[k - 1] = -(kp + 1);
            }

            /* Decrease K and return to the start of the main loop */
            k -= kstep;
        }

        /* Update the upper triangle of A11 (= A(0:k,0:k)) as
         *
         * A11 := A11 - U12*D*U12**H = A11 - U12*W**H
         *
         * (note that conjg(W) is actually stored) */
        if (k >= 0 && n - 1 - k > 0) {
            int kw_after = nb - 1 - (n - 1 - k);
            cblas_zgemmt(CblasColMajor, CblasUpper, CblasNoTrans, CblasTrans,
                         k + 1, n - 1 - k,
                         &NEG_CONE, &A[0 + (k + 1) * lda], lda,
                         &W[0 + (kw_after + 1) * ldw], ldw,
                         &CONE, &A[0], lda);
        }

        /* Put U12 in standard form by partially undoing the interchanges
         * in columns k+1:n looping backwards from k+1 to n */
        {
            int j = k + 1;
            do {
                int jj = j;
                int jp = ipiv[j];
                if (jp < 0) {
                    jp = -(jp + 1);
                    j++;
                }
                j++;
                if (jp != jj && j <= n - 1) {
                    cblas_zswap(n - j, &A[jp + j * lda], lda,
                                &A[jj + j * lda], lda);
                }
            } while (j <= n - 2);
        }

        /* Set KB to the number of columns factorized */
        *kb = n - 1 - k;

    } else {

        /* Factorize the leading columns of A using the lower triangle
         * of A and working forwards, and compute the matrix W = L21*D
         * for use in updating A22 (note that conjg(W) is actually stored)
         *
         * K is the main loop index, increasing from 1 in steps of 1 or 2. */

        int k = 0;

        while (1) {
            if ((k >= nb - 1 && nb < n) || k >= n) {
                break;
            }

            /* Copy column K of A to column K of W and update it */
            W[k + k * ldw] = creal(A[k + k * lda]);
            if (k < n - 1) {
                cblas_zcopy(n - k - 1, &A[(k + 1) + k * lda], 1,
                            &W[(k + 1) + k * ldw], 1);
            }

            if (k > 0) {
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            n - k, k,
                            &NEG_CONE, &A[k + 0 * lda], lda,
                            &W[k + 0 * ldw], ldw,
                            &CONE, &W[k + k * ldw], 1);
                W[k + k * ldw] = creal(W[k + k * ldw]);
            }

            int kstep = 1;

            /* Determine rows and columns to be interchanged and whether
             * a 1-by-1 or 2-by-2 pivot block will be used */
            double absakk = fabs(creal(W[k + k * ldw]));

            /* IMAX is the row-index of the largest off-diagonal element in
             * column K, and COLMAX is its absolute value.
             * Determine both COLMAX and IMAX. */
            int imax = k;
            double colmax = 0.0;
            if (k < n - 1) {
                imax = (k + 1) + cblas_izamax(n - k - 1, &W[(k + 1) + k * ldw], 1);
                colmax = cabs1(W[imax + k * ldw]);
            }

            int kp;

            if (fmax(absakk, colmax) == 0.0) {
                /* Column K is zero or underflow: set INFO and continue */
                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
                A[k + k * lda] = creal(A[k + k * lda]);
            } else {

                if (absakk >= ALPHA_BK * colmax) {
                    /* No interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* Copy column IMAX to column K+1 of W and update it */
                    cblas_zcopy(imax - k, &A[imax + k * lda], lda,
                                &W[k + (k + 1) * ldw], 1);
                    zlacgv(imax - k, &W[k + (k + 1) * ldw], 1);

                    W[imax + (k + 1) * ldw] = creal(A[imax + imax * lda]);

                    if (imax < n - 1) {
                        cblas_zcopy(n - imax - 1, &A[(imax + 1) + imax * lda], 1,
                                    &W[(imax + 1) + (k + 1) * ldw], 1);
                    }

                    if (k > 0) {
                        cblas_zgemv(CblasColMajor, CblasNoTrans,
                                    n - k, k,
                                    &NEG_CONE, &A[k + 0 * lda], lda,
                                    &W[imax + 0 * ldw], ldw,
                                    &CONE, &W[k + (k + 1) * ldw], 1);
                        W[imax + (k + 1) * ldw] = creal(W[imax + (k + 1) * ldw]);
                    }

                    /* JMAX is the column-index of the largest off-diagonal
                     * element in row IMAX, and ROWMAX is its absolute value */
                    int jmax = k + cblas_izamax(imax - k, &W[k + (k + 1) * ldw], 1);
                    double rowmax = cabs1(W[jmax + (k + 1) * ldw]);

                    if (imax < n - 1) {
                        jmax = (imax + 1) + cblas_izamax(n - imax - 1, &W[(imax + 1) + (k + 1) * ldw], 1);
                        rowmax = fmax(rowmax, cabs1(W[jmax + (k + 1) * ldw]));
                    }

                    if (absakk >= ALPHA_BK * colmax * (colmax / rowmax)) {
                        /* No interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (fabs(creal(W[imax + (k + 1) * ldw])) >= ALPHA_BK * rowmax) {
                        /* Interchange rows and columns K and IMAX,
                         * use 1-by-1 pivot block.
                         * Copy column K+1 of W to column K of W. */
                        kp = imax;
                        cblas_zcopy(n - k, &W[k + (k + 1) * ldw], 1,
                                    &W[k + k * ldw], 1);
                    } else {
                        /* Interchange rows and columns K+1 and IMAX,
                         * use 2-by-2 pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                /* KK is the column of A where pivoting step stopped */
                int kk = k + kstep - 1;

                /* Interchange rows and columns KP and KK.
                 * Updated column KP is already stored in column KK of W. */
                if (kp != kk) {
                    A[kp + kp * lda] = creal(A[kk + kk * lda]);

                    if (kp - kk - 1 > 0) {
                        cblas_zcopy(kp - kk - 1, &A[(kk + 1) + kk * lda], 1,
                                    &A[kp + (kk + 1) * lda], lda);
                        zlacgv(kp - kk - 1, &A[kp + (kk + 1) * lda], lda);
                    }

                    if (kp < n - 1) {
                        cblas_zcopy(n - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);
                    }

                    if (k > 0) {
                        cblas_zswap(k, &A[kk + 0 * lda], lda,
                                    &A[kp + 0 * lda], lda);
                    }

                    cblas_zswap(kk + 1, &W[kk + 0 * ldw], ldw,
                                &W[kp + 0 * ldw], ldw);
                }

                if (kstep == 1) {

                    cblas_zcopy(n - k, &W[k + k * ldw], 1, &A[k + k * lda], 1);

                    if (k < n - 1) {
                        double r1 = 1.0 / creal(A[k + k * lda]);
                        cblas_zdscal(n - k - 1, r1, &A[(k + 1) + k * lda], 1);

                        zlacgv(n - k - 1, &W[(k + 1) + k * ldw], 1);
                    }

                } else {

                    if (k < n - 2) {
                        double complex d21 = W[(k + 1) + k * ldw];
                        double complex d11 = W[(k + 1) + (k + 1) * ldw] / d21;
                        double complex d22 = W[k + k * ldw] / conj(d21);
                        double t = 1.0 / (creal(d11 * d22) - 1.0);
                        d21 = t / d21;

                        for (int j = k + 2; j <= n - 1; j++) {
                            A[j + k * lda] = conj(d21) * (d11 * W[j + k * ldw] - W[j + (k + 1) * ldw]);
                            A[j + (k + 1) * lda] = d21 * (d22 * W[j + (k + 1) * ldw] - W[j + k * ldw]);
                        }
                    }

                    A[k + k * lda] = W[k + k * ldw];
                    A[(k + 1) + k * lda] = W[(k + 1) + k * ldw];
                    A[(k + 1) + (k + 1) * lda] = W[(k + 1) + (k + 1) * ldw];

                    zlacgv(n - k - 1, &W[(k + 1) + k * ldw], 1);
                    if (k < n - 2) {
                        zlacgv(n - k - 2, &W[(k + 2) + (k + 1) * ldw], 1);
                    }
                }
            }

            /* Store details of the interchanges in IPIV */
            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(kp + 1);
                ipiv[k + 1] = -(kp + 1);
            }

            /* Increase K and return to the start of the main loop */
            k += kstep;
        }

        /* Update the lower triangle of A22 (= A(k:n-1,k:n-1)) as
         *
         * A22 := A22 - L21*D*L21**H = A22 - L21*W**H
         *
         * (note that conjg(W) is actually stored) */
        if (k < n && k > 0) {
            cblas_zgemmt(CblasColMajor, CblasLower, CblasNoTrans, CblasTrans,
                         n - k, k,
                         &NEG_CONE, &A[k + 0 * lda], lda,
                         &W[k + 0 * ldw], ldw,
                         &CONE, &A[k + k * lda], lda);
        }

        /* Put L21 in standard form by partially undoing the interchanges
         * of rows in columns 1:k-1 looping backwards from k-1 to 1 */
        {
            int j = k - 1;
            do {
                int jj = j;
                int jp = ipiv[j];
                if (jp < 0) {
                    jp = -(jp + 1);
                    j--;
                }
                j--;
                if (jp != jj && j >= 0) {
                    cblas_zswap(j + 1, &A[jp + 0 * lda], lda,
                                &A[jj + 0 * lda], lda);
                }
            } while (j >= 1);
        }

        /* Set KB to the number of columns factorized */
        *kb = k;
    }
}
