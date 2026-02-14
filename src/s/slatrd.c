/**
 * @file slatrd.c
 * @brief SLATRD reduces NB rows and columns of a real symmetric matrix to
 *        tridiagonal form.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLATRD reduces NB rows and columns of a real symmetric matrix A to
 * symmetric tridiagonal form by an orthogonal similarity
 * transformation Q**T * A * Q, and returns the matrices V and W which are
 * needed to apply the transformation to the unreduced part of A.
 *
 * If UPLO = 'U', SLATRD reduces the last NB rows and columns of a
 * matrix, of which the upper triangle is supplied;
 * if UPLO = 'L', SLATRD reduces the first NB rows and columns of a
 * matrix, of which the lower triangle is supplied.
 *
 * This is an auxiliary routine called by SSYTRD.
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
void slatrd(const char* uplo, const int n, const int nb,
            f32* restrict A, const int lda,
            f32* restrict E, f32* restrict tau,
            f32* restrict W, const int ldw)
{
    const f32 ZERO = 0.0f;
    const f32 ONE  = 1.0f;
    const f32 HALF = 0.5f;

    int ii, iw;
    f32 alpha;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Reduce last NB columns of upper triangle. */
        for (ii = n; ii >= n - nb + 1; ii--) {
            iw = ii - n + nb;

            if (ii < n) {
                /* Update A(1:ii, ii). */
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            ii, n - ii, -ONE,
                            &A[ii * lda], lda,
                            &W[(ii - 1) + iw * ldw], ldw,
                            ONE, &A[(ii - 1) * lda], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            ii, n - ii, -ONE,
                            &W[iw * ldw], ldw,
                            &A[(ii - 1) + ii * lda], lda,
                            ONE, &A[(ii - 1) * lda], 1);
            }

            if (ii > 1) {
                /* Generate elementary reflector H(ii) to annihilate A(1:ii-2, ii). */
                slarfg(ii - 1,
                       &A[(ii - 2) + (ii - 1) * lda],
                       &A[(ii - 1) * lda],
                       1,
                       &tau[ii - 2]);

                E[ii - 2] = A[(ii - 2) + (ii - 1) * lda];
                A[(ii - 2) + (ii - 1) * lda] = ONE;

                /* Compute W(1:ii-1, iw). */
                cblas_ssymv(CblasColMajor, CblasUpper,
                            ii - 1, ONE,
                            A, lda,
                            &A[(ii - 1) * lda], 1,
                            ZERO, &W[(iw - 1) * ldw], 1);

                if (ii < n) {
                    cblas_sgemv(CblasColMajor, CblasTrans,
                                ii - 1, n - ii, ONE,
                                &W[iw * ldw], ldw,
                                &A[(ii - 1) * lda], 1,
                                ZERO, &W[ii + (iw - 1) * ldw], 1);
                    cblas_sgemv(CblasColMajor, CblasNoTrans,
                                ii - 1, n - ii, -ONE,
                                &A[ii * lda], lda,
                                &W[ii + (iw - 1) * ldw], 1,
                                ONE, &W[(iw - 1) * ldw], 1);
                    cblas_sgemv(CblasColMajor, CblasTrans,
                                ii - 1, n - ii, ONE,
                                &A[ii * lda], lda,
                                &A[(ii - 1) * lda], 1,
                                ZERO, &W[ii + (iw - 1) * ldw], 1);
                    cblas_sgemv(CblasColMajor, CblasNoTrans,
                                ii - 1, n - ii, -ONE,
                                &W[iw * ldw], ldw,
                                &W[ii + (iw - 1) * ldw], 1,
                                ONE, &W[(iw - 1) * ldw], 1);
                }

                cblas_sscal(ii - 1, tau[ii - 2], &W[(iw - 1) * ldw], 1);
                alpha = -HALF * tau[ii - 2] *
                        cblas_sdot(ii - 1,
                                   &W[(iw - 1) * ldw], 1,
                                   &A[(ii - 1) * lda], 1);
                cblas_saxpy(ii - 1, alpha,
                            &A[(ii - 1) * lda], 1,
                            &W[(iw - 1) * ldw], 1);
            }
        }

    } else {
        /* Reduce first NB columns of lower triangle. */
        for (ii = 1; ii <= nb; ii++) {
            /* Update A(ii:n, ii). */
            cblas_sgemv(CblasColMajor, CblasNoTrans,
                        n - ii + 1, ii - 1, -ONE,
                        &A[ii - 1], lda,
                        &W[ii - 1], ldw,
                        ONE, &A[(ii - 1) + (ii - 1) * lda], 1);
            cblas_sgemv(CblasColMajor, CblasNoTrans,
                        n - ii + 1, ii - 1, -ONE,
                        &W[ii - 1], ldw,
                        &A[ii - 1], lda,
                        ONE, &A[(ii - 1) + (ii - 1) * lda], 1);

            if (ii < n) {
                /* Generate elementary reflector H(ii) to annihilate A(ii+2:n, ii). */
                int min_row = (ii + 1 < n - 1) ? (ii + 1) : (n - 1);
                slarfg(n - ii,
                       &A[ii + (ii - 1) * lda],
                       &A[min_row + (ii - 1) * lda],
                       1,
                       &tau[ii - 1]);

                E[ii - 1] = A[ii + (ii - 1) * lda];
                A[ii + (ii - 1) * lda] = ONE;

                /* Compute W(ii+1:n, ii). */
                cblas_ssymv(CblasColMajor, CblasLower,
                            n - ii, ONE,
                            &A[ii + ii * lda], lda,
                            &A[ii + (ii - 1) * lda], 1,
                            ZERO, &W[ii + (ii - 1) * ldw], 1);
                cblas_sgemv(CblasColMajor, CblasTrans,
                            n - ii, ii - 1, ONE,
                            &W[ii], ldw,
                            &A[ii + (ii - 1) * lda], 1,
                            ZERO, &W[(ii - 1) * ldw], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            n - ii, ii - 1, -ONE,
                            &A[ii], lda,
                            &W[(ii - 1) * ldw], 1,
                            ONE, &W[ii + (ii - 1) * ldw], 1);
                cblas_sgemv(CblasColMajor, CblasTrans,
                            n - ii, ii - 1, ONE,
                            &A[ii], lda,
                            &A[ii + (ii - 1) * lda], 1,
                            ZERO, &W[(ii - 1) * ldw], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            n - ii, ii - 1, -ONE,
                            &W[ii], ldw,
                            &W[(ii - 1) * ldw], 1,
                            ONE, &W[ii + (ii - 1) * ldw], 1);

                cblas_sscal(n - ii, tau[ii - 1], &W[ii + (ii - 1) * ldw], 1);
                alpha = -HALF * tau[ii - 1] *
                        cblas_sdot(n - ii,
                                   &W[ii + (ii - 1) * ldw], 1,
                                   &A[ii + (ii - 1) * lda], 1);
                cblas_saxpy(n - ii, alpha,
                            &A[ii + (ii - 1) * lda], 1,
                            &W[ii + (ii - 1) * ldw], 1);
            }
        }
    }
}
