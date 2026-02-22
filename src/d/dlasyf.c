/**
 * @file dlasyf.c
 * @brief DLASYF computes a partial factorization of a real symmetric matrix
 *        using the Bunch-Kaufman diagonal pivoting method.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/* Alpha for Bunch-Kaufman pivoting: (1 + sqrt(17)) / 8 */
static const f64 ALPHA_BK = 0.6403882032022076;

/**
 * DLASYF computes a partial factorization of a real symmetric matrix A
 * using the Bunch-Kaufman diagonal pivoting method. The partial
 * factorization has the form:
 *
 *    A = ( I  U12 ) ( A11  0  ) (  I       0    )  if UPLO = 'U', or:
 *        ( 0  U22 ) (  0   D  ) ( U12**T U22**T )
 *
 *    A = ( L11  0 ) (  D   0  ) ( L11**T L21**T )  if UPLO = 'L'
 *        ( L21  I ) (  0  A22 ) (  0       I    )
 *
 * where the order of D is at most NB. The actual order is returned in
 * the argument KB, and is either NB or NB-1, or N if N <= NB.
 *
 * DLASYF is an auxiliary routine called by DSYTRF. It uses blocked code
 * (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = 'U') or
 * A22 (if UPLO = 'L').
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the symmetric matrix A is stored:
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nb    The maximum number of columns of the matrix A that
 *                      should be factored. nb should be at least 2 to allow
 *                      for 2-by-2 pivot blocks.
 * @param[out]    kb    The number of columns of A that were actually factored.
 *                      kb is either nb-1 or nb, or n if n <= nb.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the symmetric matrix A. If uplo = 'U', the
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
 * @param[out]    W     Double precision array, dimension (ldw, nb).
 *                      Workspace for storing updated columns during factorization.
 * @param[in]     ldw   The leading dimension of the array W. ldw >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - > 0: if info = k+1, D(k,k) is exactly zero. The
 *                           factorization has been completed, but the block
 *                           diagonal matrix D is exactly singular.
 */
void dlasyf(
    const char* uplo,
    const INT n,
    const INT nb,
    INT* kb,
    f64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    f64* restrict W,
    const INT ldw,
    INT* info)
{
    *info = 0;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /* Factorize the trailing columns of A using the upper triangle
         * of A and working backwards, and compute the matrix W = U12*D
         * for use in updating A11.
         *
         * K is the main loop index, decreasing from N in steps of 1 or 2.
         *
         * KW is the column of W which corresponds to column K of A. */

        INT k = n - 1;

        while (1) {
            INT kw = nb - 1 - (n - 1 - k);

            if ((k <= n - nb && nb < n) || k < 0) {
                break;
            }

            /* Copy column K of A to column KW of W and update it */
            cblas_dcopy(k + 1, &A[0 + k * lda], 1, &W[0 + kw * ldw], 1);

            if (k < n - 1) {
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            k + 1, n - 1 - k,
                            -1.0, &A[0 + (k + 1) * lda], lda,
                            &W[k + (kw + 1) * ldw], ldw,
                            1.0, &W[0 + kw * ldw], 1);
            }

            INT kstep = 1;

            /* Determine rows and columns to be interchanged and whether
             * a 1-by-1 or 2-by-2 pivot block will be used */
            f64 absakk = fabs(W[k + kw * ldw]);

            /* IMAX is the row-index of the largest off-diagonal element in
             * column K, and COLMAX is its absolute value.
             * Determine both COLMAX and IMAX. */
            INT imax = 0;
            f64 colmax = 0.0;
            if (k > 0) {
                imax = cblas_idamax(k, &W[0 + kw * ldw], 1);
                colmax = fabs(W[imax + kw * ldw]);
            }

            INT kp;

            if (fmax(absakk, colmax) == 0.0) {
                /* Column K is zero or underflow: set INFO and continue */
                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
            } else {
                if (absakk >= ALPHA_BK * colmax) {
                    /* No interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* Copy column IMAX to column KW-1 of W and update it */
                    cblas_dcopy(imax + 1, &A[0 + imax * lda], 1,
                                &W[0 + (kw - 1) * ldw], 1);

                    if (k - imax > 0) {
                        cblas_dcopy(k - imax, &A[imax + (imax + 1) * lda], lda,
                                    &W[(imax + 1) + (kw - 1) * ldw], 1);
                    }

                    if (k < n - 1) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans,
                                    k + 1, n - 1 - k,
                                    -1.0, &A[0 + (k + 1) * lda], lda,
                                    &W[imax + (kw + 1) * ldw], ldw,
                                    1.0, &W[0 + (kw - 1) * ldw], 1);
                    }

                    /* JMAX is the column-index of the largest off-diagonal
                     * element in row IMAX, and ROWMAX is its absolute value */
                    INT jmax = (imax + 1) + cblas_idamax(k - imax, &W[(imax + 1) + (kw - 1) * ldw], 1);
                    f64 rowmax = fabs(W[jmax + (kw - 1) * ldw]);

                    if (imax > 0) {
                        jmax = cblas_idamax(imax, &W[0 + (kw - 1) * ldw], 1);
                        rowmax = fmax(rowmax, fabs(W[jmax + (kw - 1) * ldw]));
                    }

                    if (absakk >= ALPHA_BK * colmax * (colmax / rowmax)) {
                        /* No interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (fabs(W[imax + (kw - 1) * ldw]) >= ALPHA_BK * rowmax) {
                        /* Interchange rows and columns K and IMAX,
                         * use 1-by-1 pivot block.
                         * Copy column KW-1 of W to column KW of W. */
                        kp = imax;
                        cblas_dcopy(k + 1, &W[0 + (kw - 1) * ldw], 1,
                                    &W[0 + kw * ldw], 1);
                    } else {
                        /* Interchange rows and columns K-1 and IMAX,
                         * use 2-by-2 pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                /* KK is the column of A where pivoting step stopped */
                INT kk = k - kstep + 1;

                /* KKW is the column of W which corresponds to column KK of A */
                INT kkw = nb - 1 - (n - 1 - kk);

                /* Interchange rows and columns KP and KK.
                 * Updated column KP is already stored in column KKW of W. */
                if (kp != kk) {
                    /* Copy non-updated column KK to column KP of submatrix A */
                    A[kp + kp * lda] = A[kk + kk * lda];

                    if (kk - kp - 1 > 0) {
                        cblas_dcopy(kk - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[kp + (kp + 1) * lda], lda);
                    }

                    if (kp > 0) {
                        cblas_dcopy(kp, &A[0 + kk * lda], 1,
                                    &A[0 + kp * lda], 1);
                    }

                    /* Interchange rows KK and KP in last K+1 to N columns of A.
                     * Interchange rows KK and KP in last KKW to NB columns of W. */
                    if (k < n - 1) {
                        cblas_dswap(n - 1 - k, &A[kk + (k + 1) * lda], lda,
                                    &A[kp + (k + 1) * lda], lda);
                    }

                    cblas_dswap(nb - kkw, &W[kk + kkw * ldw], ldw,
                                &W[kp + kkw * ldw], ldw);
                }

                if (kstep == 1) {
                    /* 1-by-1 pivot block D(k): column KW of W now holds
                     *
                     * W(kw) = U(k)*D(k),
                     *
                     * where U(k) is the k-th column of U.
                     *
                     * Store subdiag. elements of column U(k)
                     * and 1-by-1 block D(k) in column k of A.
                     * NOTE: Diagonal element U(k,k) is a UNIT element
                     * and not stored.
                     *    A(k,k) := D(k,k) = W(k,kw)
                     *    A(0:k-1,k) := U(0:k-1,k) = W(0:k-1,kw)/D(k,k) */
                    cblas_dcopy(k + 1, &W[0 + kw * ldw], 1, &A[0 + k * lda], 1);

                    if (k > 0) {
                        f64 r1 = 1.0 / A[k + k * lda];
                        cblas_dscal(k, r1, &A[0 + k * lda], 1);
                    }
                } else {
                    /* 2-by-2 pivot block D(k): columns KW and KW-1 of W now hold
                     *
                     * ( W(kw-1) W(kw) ) = ( U(k-1) U(k) )*D(k)
                     *
                     * where U(k) and U(k-1) are the k-th and (k-1)-th columns of U.
                     *
                     * Store U(0:k-2,k-1) and U(0:k-2,k) and 2-by-2
                     * block D(k-1:k,k-1:k) in columns k-1 and k of A.
                     * NOTE: 2-by-2 diagonal block U(k-1:k,k-1:k) is a UNIT
                     * block and not stored.
                     *    A(k-1:k,k-1:k) := D(k-1:k,k-1:k) = W(k-1:k,kw-1:kw)
                     *    A(0:k-2,k-1:k) := U(0:k-2,k-1:k) =
                     *    = W(0:k-2,kw-1:kw) * ( D(k-1:k,k-1:k)**(-1) ) */
                    if (k > 1) {
                        /* Compose the columns of the inverse of 2-by-2 pivot
                         * block D in the following way to reduce the number
                         * of FLOPS when we myltiply panel ( W(kw-1) W(kw) ) by
                         * this inverse
                         *
                         * D**(-1) = ( d11 d21 )**(-1) =
                         *           ( d21 d22 )
                         *
                         * = 1/(d11*d22-d21**2) * ( ( d22 ) (-d21 ) ) =
                         *                        ( (-d21 ) ( d11 ) )
                         *
                         * = 1/d21 * 1/((d11/d21)*(d22/d21)-1) *
                         *
                         *   * ( ( d22/d21 ) (      -1 ) ) =
                         *     ( (      -1 ) ( d11/d21 ) )
                         *
                         * = 1/d21 * 1/(D22*D11-1) * ( ( D11 ) (  -1 ) ) =
                         *                           ( ( -1  ) ( D22 ) )
                         *
                         * = 1/d21 * T * ( ( D11 ) (  -1 ) )
                         *               ( (  -1 ) ( D22 ) )
                         *
                         * = D21 * ( ( D11 ) (  -1 ) )
                         *         ( (  -1 ) ( D22 ) ) */
                        f64 d21 = W[(k - 1) + kw * ldw];
                        f64 d11 = W[k + kw * ldw] / d21;
                        f64 d22 = W[(k - 1) + (kw - 1) * ldw] / d21;
                        f64 t = 1.0 / (d11 * d22 - 1.0);
                        d21 = t / d21;

                        /* Update elements in columns A(k-1) and A(k) as
                         * dot products of rows of ( W(kw-1) W(kw) ) and columns
                         * of D**(-1) */
                        for (INT j = 0; j <= k - 2; j++) {
                            A[j + (k - 1) * lda] = d21 * (d11 * W[j + (kw - 1) * ldw] - W[j + kw * ldw]);
                            A[j + k * lda] = d21 * (d22 * W[j + kw * ldw] - W[j + (kw - 1) * ldw]);
                        }
                    }

                    /* Copy D(k) to A */
                    A[(k - 1) + (k - 1) * lda] = W[(k - 1) + (kw - 1) * ldw];
                    A[(k - 1) + k * lda] = W[(k - 1) + kw * ldw];
                    A[k + k * lda] = W[k + kw * ldw];
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
         * A11 := A11 - U12*D*U12**T = A11 - U12*W**T */
        if (k >= 0 && n - 1 - k > 0) {
            INT kw_after = nb - 1 - (n - 1 - k);
            cblas_dgemmt(CblasColMajor, CblasUpper, CblasNoTrans, CblasTrans,
                         k + 1, n - 1 - k,
                         -1.0, &A[0 + (k + 1) * lda], lda,
                         &W[0 + (kw_after + 1) * ldw], ldw,
                         1.0, &A[0], lda);
        }

        /* Put U12 in standard form by partially undoing the interchanges
         * in columns k+1:n looping backwards from k+1 to n */
        {
            INT j = k + 1;
            do {
                INT jj = j;
                INT jp = ipiv[j];
                if (jp < 0) {
                    jp = -(jp + 1);
                    j++;
                }
                j++;
                if (jp != jj && j <= n - 1) {
                    cblas_dswap(n - j, &A[jp + j * lda], lda,
                                &A[jj + j * lda], lda);
                }
            } while (j <= n - 2);
        }

        /* Set KB to the number of columns factorized */
        *kb = n - 1 - k;

    } else {

        /* Factorize the leading columns of A using the lower triangle
         * of A and working forwards, and compute the matrix W = L21*D
         * for use in updating A22.
         *
         * K is the main loop index, increasing from 1 in steps of 1 or 2. */

        INT k = 0;

        while (1) {
            if ((k >= nb - 1 && nb < n) || k >= n) {
                break;
            }

            /* Copy column K of A to column K of W and update it */
            cblas_dcopy(n - k, &A[k + k * lda], 1, &W[k + k * ldw], 1);

            if (k > 0) {
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            n - k, k,
                            -1.0, &A[k + 0 * lda], lda,
                            &W[k + 0 * ldw], ldw,
                            1.0, &W[k + k * ldw], 1);
            }

            INT kstep = 1;

            /* Determine rows and columns to be interchanged and whether
             * a 1-by-1 or 2-by-2 pivot block will be used */
            f64 absakk = fabs(W[k + k * ldw]);

            /* IMAX is the row-index of the largest off-diagonal element in
             * column K, and COLMAX is its absolute value.
             * Determine both COLMAX and IMAX. */
            INT imax = k;
            f64 colmax = 0.0;
            if (k < n - 1) {
                imax = (k + 1) + cblas_idamax(n - k - 1, &W[(k + 1) + k * ldw], 1);
                colmax = fabs(W[imax + k * ldw]);
            }

            INT kp;

            if (fmax(absakk, colmax) == 0.0) {
                /* Column K is zero or underflow: set INFO and continue */
                if (*info == 0) {
                    *info = k + 1;
                }
                kp = k;
            } else {
                if (absakk >= ALPHA_BK * colmax) {
                    /* No interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /* Copy column IMAX to column K+1 of W and update it */
                    cblas_dcopy(imax - k, &A[imax + k * lda], lda,
                                &W[k + (k + 1) * ldw], 1);

                    cblas_dcopy(n - imax, &A[imax + imax * lda], 1,
                                &W[imax + (k + 1) * ldw], 1);

                    if (k > 0) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans,
                                    n - k, k,
                                    -1.0, &A[k + 0 * lda], lda,
                                    &W[imax + 0 * ldw], ldw,
                                    1.0, &W[k + (k + 1) * ldw], 1);
                    }

                    /* JMAX is the column-index of the largest off-diagonal
                     * element in row IMAX, and ROWMAX is its absolute value */
                    INT jmax = k + cblas_idamax(imax - k, &W[k + (k + 1) * ldw], 1);
                    f64 rowmax = fabs(W[jmax + (k + 1) * ldw]);

                    if (imax < n - 1) {
                        jmax = (imax + 1) + cblas_idamax(n - imax - 1, &W[(imax + 1) + (k + 1) * ldw], 1);
                        rowmax = fmax(rowmax, fabs(W[jmax + (k + 1) * ldw]));
                    }

                    if (absakk >= ALPHA_BK * colmax * (colmax / rowmax)) {
                        /* No interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (fabs(W[imax + (k + 1) * ldw]) >= ALPHA_BK * rowmax) {
                        /* Interchange rows and columns K and IMAX,
                         * use 1-by-1 pivot block.
                         * Copy column K+1 of W to column K of W. */
                        kp = imax;
                        cblas_dcopy(n - k, &W[k + (k + 1) * ldw], 1,
                                    &W[k + k * ldw], 1);
                    } else {
                        /* Interchange rows and columns K+1 and IMAX,
                         * use 2-by-2 pivot block */
                        kp = imax;
                        kstep = 2;
                    }
                }

                /* KK is the column of A where pivoting step stopped */
                INT kk = k + kstep - 1;

                /* Interchange rows and columns KP and KK.
                 * Updated column KP is already stored in column KK of W. */
                if (kp != kk) {
                    /* Copy non-updated column KK to column KP of submatrix A */
                    A[kp + kp * lda] = A[kk + kk * lda];

                    if (kp - kk - 1 > 0) {
                        cblas_dcopy(kp - kk - 1, &A[(kk + 1) + kk * lda], 1,
                                    &A[kp + (kk + 1) * lda], lda);
                    }

                    if (kp < n - 1) {
                        cblas_dcopy(n - kp - 1, &A[(kp + 1) + kk * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);
                    }

                    /* Interchange rows KK and KP in first K-1 columns of A.
                     * Interchange rows KK and KP in first KK columns of W. */
                    if (k > 0) {
                        cblas_dswap(k, &A[kk + 0 * lda], lda,
                                    &A[kp + 0 * lda], lda);
                    }

                    cblas_dswap(kk + 1, &W[kk + 0 * ldw], ldw,
                                &W[kp + 0 * ldw], ldw);
                }

                if (kstep == 1) {
                    /* 1-by-1 pivot block D(k): column k of W now holds
                     *
                     * W(k) = L(k)*D(k),
                     *
                     * where L(k) is the k-th column of L.
                     *
                     * Store subdiag. elements of column L(k)
                     * and 1-by-1 block D(k) in column k of A.
                     * (NOTE: Diagonal element L(k,k) is a UNIT element
                     * and not stored)
                     *    A(k,k) := D(k,k) = W(k,k)
                     *    A(k+1:n-1,k) := L(k+1:n-1,k) = W(k+1:n-1,k)/D(k,k) */
                    cblas_dcopy(n - k, &W[k + k * ldw], 1, &A[k + k * lda], 1);

                    if (k < n - 1) {
                        f64 r1 = 1.0 / A[k + k * lda];
                        cblas_dscal(n - k - 1, r1, &A[(k + 1) + k * lda], 1);
                    }
                } else {
                    /* 2-by-2 pivot block D(k): columns k and k+1 of W now hold
                     *
                     * ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
                     *
                     * where L(k) and L(k+1) are the k-th and (k+1)-th columns of L.
                     *
                     * Store L(k+2:n-1,k) and L(k+2:n-1,k+1) and 2-by-2
                     * block D(k:k+1,k:k+1) in columns k and k+1 of A.
                     * (NOTE: 2-by-2 diagonal block L(k:k+1,k:k+1) is a UNIT
                     * block and not stored)
                     *    A(k:k+1,k:k+1) := D(k:k+1,k:k+1) = W(k:k+1,k:k+1)
                     *    A(k+2:n-1,k:k+1) := L(k+2:n-1,k:k+1) =
                     *    = W(k+2:n-1,k:k+1) * ( D(k:k+1,k:k+1)**(-1) ) */
                    if (k < n - 2) {
                        /* Compose the columns of the inverse of 2-by-2 pivot
                         * block D in the following way to reduce the number
                         * of FLOPS when we myltiply panel ( W(k) W(k+1) ) by
                         * this inverse
                         *
                         * D**(-1) = ( d11 d21 )**(-1) =
                         *           ( d21 d22 )
                         *
                         * = 1/(d11*d22-d21**2) * ( ( d22 ) (-d21 ) ) =
                         *                        ( (-d21 ) ( d11 ) )
                         *
                         * = 1/d21 * 1/((d11/d21)*(d22/d21)-1) *
                         *
                         *   * ( ( d22/d21 ) (      -1 ) ) =
                         *     ( (      -1 ) ( d11/d21 ) )
                         *
                         * = 1/d21 * 1/(D22*D11-1) * ( ( D11 ) (  -1 ) ) =
                         *                           ( ( -1  ) ( D22 ) )
                         *
                         * = 1/d21 * T * ( ( D11 ) (  -1 ) )
                         *               ( (  -1 ) ( D22 ) )
                         *
                         * = D21 * ( ( D11 ) (  -1 ) )
                         *         ( (  -1 ) ( D22 ) ) */
                        f64 d21 = W[(k + 1) + k * ldw];
                        f64 d11 = W[(k + 1) + (k + 1) * ldw] / d21;
                        f64 d22 = W[k + k * ldw] / d21;
                        f64 t = 1.0 / (d11 * d22 - 1.0);
                        d21 = t / d21;

                        /* Update elements in columns A(k) and A(k+1) as
                         * dot products of rows of ( W(k) W(k+1) ) and columns
                         * of D**(-1) */
                        for (INT j = k + 2; j <= n - 1; j++) {
                            A[j + k * lda] = d21 * (d11 * W[j + k * ldw] - W[j + (k + 1) * ldw]);
                            A[j + (k + 1) * lda] = d21 * (d22 * W[j + (k + 1) * ldw] - W[j + k * ldw]);
                        }
                    }

                    /* Copy D(k) to A */
                    A[k + k * lda] = W[k + k * ldw];
                    A[(k + 1) + k * lda] = W[(k + 1) + k * ldw];
                    A[(k + 1) + (k + 1) * lda] = W[(k + 1) + (k + 1) * ldw];
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
         * A22 := A22 - L21*D*L21**T = A22 - L21*W**T */
        if (k < n && k > 0) {
            cblas_dgemmt(CblasColMajor, CblasLower, CblasNoTrans, CblasTrans,
                         n - k, k,
                         -1.0, &A[k + 0 * lda], lda,
                         &W[k + 0 * ldw], ldw,
                         1.0, &A[k + k * lda], lda);
        }

        /* Put L21 in standard form by partially undoing the interchanges
         * of rows in columns 1:k-1 looping backwards from k-1 to 1 */
        {
            INT j = k - 1;
            do {
                INT jj = j;
                INT jp = ipiv[j];
                if (jp < 0) {
                    jp = -(jp + 1);
                    j--;
                }
                j--;
                if (jp != jj && j >= 0) {
                    cblas_dswap(j + 1, &A[jp + 0 * lda], lda,
                                &A[jj + 0 * lda], lda);
                }
            } while (j >= 1);
        }

        /* Set KB to the number of columns factorized */
        *kb = k;
    }
}
