/**
 * @file dlatrd.c
 * @brief DLATRD reduces NB rows and columns of a real symmetric matrix to
 *        tridiagonal form.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLATRD reduces NB rows and columns of a real symmetric matrix A to
 * symmetric tridiagonal form by an orthogonal similarity
 * transformation Q**T * A * Q, and returns the matrices V and W which are
 * needed to apply the transformation to the unreduced part of A.
 *
 * If UPLO = 'U', DLATRD reduces the last NB rows and columns of a
 * matrix, of which the upper triangle is supplied;
 * if UPLO = 'L', DLATRD reduces the first NB rows and columns of a
 * matrix, of which the lower triangle is supplied.
 *
 * This is an auxiliary routine called by DSYTRD.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the symmetric matrix A is stored:
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nb    The number of rows and columns to be reduced.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the symmetric matrix A.
 *                      On exit, if UPLO = 'U', the last NB columns have been
 *                      reduced to tridiagonal form; if UPLO = 'L', the first
 *                      NB columns have been reduced to tridiagonal form.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[out]    E     Double precision array, dimension (n-1).
 *                      The off-diagonal elements of the tridiagonal matrix.
 * @param[out]    tau   Double precision array, dimension (n-1).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    W     Double precision array, dimension (ldw, nb).
 *                      The n-by-nb matrix W.
 * @param[in]     ldw   The leading dimension of the array W. ldw >= max(1,n).
 */
void dlatrd(const char* uplo, const INT n, const INT nb,
            f64* restrict A, const INT lda,
            f64* restrict E, f64* restrict tau,
            f64* restrict W, const INT ldw)
{
    const f64 ZERO = 0.0;
    const f64 ONE  = 1.0;
    const f64 HALF = 0.5;

    INT i, iw;
    f64 alpha;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Reduce last NB columns of upper triangle. */
        for (i = n - 1; i >= n - nb; i--) {
            iw = i - n + nb;

            if (i < n - 1) {
                /* Update A(0:i, i). */
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            i + 1, n - i - 1, -ONE,
                            &A[(i + 1) * lda], lda,
                            &W[i + (iw + 1) * ldw], ldw,
                            ONE, &A[i * lda], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            i + 1, n - i - 1, -ONE,
                            &W[(iw + 1) * ldw], ldw,
                            &A[i + (i + 1) * lda], lda,
                            ONE, &A[i * lda], 1);
            }

            if (i > 0) {
                /* Generate elementary reflector H(i) to annihilate A(0:i-2, i). */
                dlarfg(i,
                       &A[(i - 1) + i * lda],
                       &A[i * lda],
                       1,
                       &tau[i - 1]);

                E[i - 1] = A[(i - 1) + i * lda];
                A[(i - 1) + i * lda] = ONE;

                /* Compute W(0:i-1, iw). */
                cblas_dsymv(CblasColMajor, CblasUpper,
                            i, ONE,
                            A, lda,
                            &A[i * lda], 1,
                            ZERO, &W[iw * ldw], 1);

                if (i < n - 1) {
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                i, n - i - 1, ONE,
                                &W[(iw + 1) * ldw], ldw,
                                &A[i * lda], 1,
                                ZERO, &W[(i + 1) + iw * ldw], 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans,
                                i, n - i - 1, -ONE,
                                &A[(i + 1) * lda], lda,
                                &W[(i + 1) + iw * ldw], 1,
                                ONE, &W[iw * ldw], 1);
                    cblas_dgemv(CblasColMajor, CblasTrans,
                                i, n - i - 1, ONE,
                                &A[(i + 1) * lda], lda,
                                &A[i * lda], 1,
                                ZERO, &W[(i + 1) + iw * ldw], 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans,
                                i, n - i - 1, -ONE,
                                &W[(iw + 1) * ldw], ldw,
                                &W[(i + 1) + iw * ldw], 1,
                                ONE, &W[iw * ldw], 1);
                }

                cblas_dscal(i, tau[i - 1], &W[iw * ldw], 1);
                alpha = -HALF * tau[i - 1] *
                        cblas_ddot(i,
                                   &W[iw * ldw], 1,
                                   &A[i * lda], 1);
                cblas_daxpy(i, alpha,
                            &A[i * lda], 1,
                            &W[iw * ldw], 1);
            }
        }

    } else {
        /* Reduce first NB columns of lower triangle. */
        for (i = 0; i < nb; i++) {
            /* Update A(i:n-1, i). */
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        n - i, i, -ONE,
                        &A[i], lda,
                        &W[i], ldw,
                        ONE, &A[i + i * lda], 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        n - i, i, -ONE,
                        &W[i], ldw,
                        &A[i], lda,
                        ONE, &A[i + i * lda], 1);

            if (i < n - 1) {
                /* Generate elementary reflector H(i) to annihilate A(i+2:n-1, i). */
                INT min_row = (i + 2 < n - 1) ? (i + 2) : (n - 1);
                dlarfg(n - i - 1,
                       &A[(i + 1) + i * lda],
                       &A[min_row + i * lda],
                       1,
                       &tau[i]);

                E[i] = A[(i + 1) + i * lda];
                A[(i + 1) + i * lda] = ONE;

                /* Compute W(i+1:n-1, i). */
                cblas_dsymv(CblasColMajor, CblasLower,
                            n - i - 1, ONE,
                            &A[(i + 1) + (i + 1) * lda], lda,
                            &A[(i + 1) + i * lda], 1,
                            ZERO, &W[(i + 1) + i * ldw], 1);
                cblas_dgemv(CblasColMajor, CblasTrans,
                            n - i - 1, i, ONE,
                            &W[i + 1], ldw,
                            &A[(i + 1) + i * lda], 1,
                            ZERO, &W[i * ldw], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            n - i - 1, i, -ONE,
                            &A[i + 1], lda,
                            &W[i * ldw], 1,
                            ONE, &W[(i + 1) + i * ldw], 1);
                cblas_dgemv(CblasColMajor, CblasTrans,
                            n - i - 1, i, ONE,
                            &A[i + 1], lda,
                            &A[(i + 1) + i * lda], 1,
                            ZERO, &W[i * ldw], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            n - i - 1, i, -ONE,
                            &W[i + 1], ldw,
                            &W[i * ldw], 1,
                            ONE, &W[(i + 1) + i * ldw], 1);

                cblas_dscal(n - i - 1, tau[i], &W[(i + 1) + i * ldw], 1);
                alpha = -HALF * tau[i] *
                        cblas_ddot(n - i - 1,
                                   &W[(i + 1) + i * ldw], 1,
                                   &A[(i + 1) + i * lda], 1);
                cblas_daxpy(n - i - 1, alpha,
                            &A[(i + 1) + i * lda], 1,
                            &W[(i + 1) + i * ldw], 1);
            }
        }
    }
}
