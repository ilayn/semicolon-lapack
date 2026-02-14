/**
 * @file slabrd.c
 * @brief SLABRD reduces the first nb rows and columns of a general matrix to bidiagonal form.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>

/**
 * SLABRD reduces the first NB rows and columns of a real general
 * m by n matrix A to upper or lower bidiagonal form by an orthogonal
 * transformation Q**T * A * P, and returns the matrices X and Y which
 * are needed to apply the transformation to the unreduced part of A.
 *
 * If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
 * bidiagonal form.
 *
 * This is an auxiliary routine called by SGEBRD.
 *
 * @param[in]     m     The number of rows in the matrix A.
 * @param[in]     n     The number of columns in the matrix A.
 * @param[in]     nb    The number of leading rows and columns of A to be reduced.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the m by n general matrix to be reduced.
 *                      On exit, the first NB rows and columns of the matrix are
 *                      overwritten; the rest of the array is unchanged.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    D     Double precision array, dimension (nb).
 *                      The diagonal elements of the first NB rows and columns of
 *                      the reduced matrix. D[i] = A[i,i].
 * @param[out]    E     Double precision array, dimension (nb).
 *                      The off-diagonal elements of the first NB rows and columns of
 *                      the reduced matrix.
 * @param[out]    tauq  Double precision array, dimension (nb).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the orthogonal matrix Q.
 * @param[out]    taup  Double precision array, dimension (nb).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the orthogonal matrix P.
 * @param[out]    X     Double precision array, dimension (ldx, nb).
 *                      The m-by-nb matrix X required to update the unreduced part
 *                      of A.
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1,m).
 * @param[out]    Y     Double precision array, dimension (ldy, nb).
 *                      The n-by-nb matrix Y required to update the unreduced part
 *                      of A.
 * @param[in]     ldy   The leading dimension of the array Y. ldy >= max(1,n).
 */
void slabrd(const int m, const int n, const int nb,
            f32* const restrict A, const int lda,
            f32* const restrict D, f32* const restrict E,
            f32* const restrict tauq, f32* const restrict taup,
            f32* const restrict X, const int ldx,
            f32* const restrict Y, const int ldy)
{
    int i;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i:m-1, i] */
            cblas_sgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        -1.0f, &A[i], lda, &Y[i], ldy, 1.0f, &A[i + i * lda], 1);
            cblas_sgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        -1.0f, &X[i], ldx, &A[i * lda], 1, 1.0f, &A[i + i * lda], 1);

            /* Generate reflection Q(i) to annihilate A[i+1:m-1, i] */
            slarfg(m - i, &A[i + i * lda], &A[((i + 1) < m ? (i + 1) : (m - 1)) + i * lda], 1,
                   &tauq[i]);
            D[i] = A[i + i * lda];

            if (i < n - 1) {
                A[i + i * lda] = 1.0f;

                /* Compute Y[i+1:n-1, i] */
                cblas_sgemv(CblasColMajor, CblasTrans, m - i, n - i - 1,
                            1.0f, &A[i + (i + 1) * lda], lda,
                            &A[i + i * lda], 1, 0.0f, &Y[i + 1 + i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, m - i, i,
                            1.0f, &A[i], lda, &A[i + i * lda], 1, 0.0f, &Y[i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            -1.0f, &Y[i + 1], ldy, &Y[i * ldy], 1, 1.0f, &Y[i + 1 + i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, m - i, i,
                            1.0f, &X[i], ldx, &A[i + i * lda], 1, 0.0f, &Y[i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, i, n - i - 1,
                            -1.0f, &A[(i + 1) * lda], lda, &Y[i * ldy], 1, 1.0f, &Y[i + 1 + i * ldy], 1);
                cblas_sscal(n - i - 1, tauq[i], &Y[i + 1 + i * ldy], 1);

                /* Update A[i, i+1:n-1] */
                cblas_sgemv(CblasColMajor, CblasNoTrans, n - i - 1, i + 1,
                            -1.0f, &Y[i + 1], ldy, &A[i], lda, 1.0f, &A[i + (i + 1) * lda], lda);
                cblas_sgemv(CblasColMajor, CblasTrans, i, n - i - 1,
                            -1.0f, &A[(i + 1) * lda], lda, &X[i], ldx, 1.0f, &A[i + (i + 1) * lda], lda);

                /* Generate reflection P(i) to annihilate A[i, i+2:n-1] */
                slarfg(n - i - 1, &A[i + (i + 1) * lda],
                       &A[i + ((i + 2) < n ? (i + 2) : (n - 1)) * lda], lda, &taup[i]);
                E[i] = A[i + (i + 1) * lda];
                A[i + (i + 1) * lda] = 1.0f;

                /* Compute X[i+1:m-1, i] */
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i - 1,
                            1.0f, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda, 0.0f, &X[i + 1 + i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, n - i - 1, i + 1,
                            1.0f, &Y[i + 1], ldy, &A[i + (i + 1) * lda], lda, 0.0f, &X[i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            -1.0f, &A[i + 1], lda, &X[i * ldx], 1, 1.0f, &X[i + 1 + i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, i, n - i - 1,
                            1.0f, &A[(i + 1) * lda], lda, &A[i + (i + 1) * lda], lda, 0.0f, &X[i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0f, &X[i + 1], ldx, &X[i * ldx], 1, 1.0f, &X[i + 1 + i * ldx], 1);
                cblas_sscal(m - i - 1, taup[i], &X[i + 1 + i * ldx], 1);
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i, i:n-1] */
            cblas_sgemv(CblasColMajor, CblasNoTrans, n - i, i,
                        -1.0f, &Y[i], ldy, &A[i], lda, 1.0f, &A[i + i * lda], lda);
            cblas_sgemv(CblasColMajor, CblasTrans, i, n - i,
                        -1.0f, &A[i * lda], lda, &X[i], ldx, 1.0f, &A[i + i * lda], lda);

            /* Generate reflection P(i) to annihilate A[i, i+1:n-1] */
            slarfg(n - i, &A[i + i * lda], &A[i + ((i + 1) < n ? (i + 1) : (n - 1)) * lda], lda,
                   &taup[i]);
            D[i] = A[i + i * lda];

            if (i < m - 1) {
                A[i + i * lda] = 1.0f;

                /* Compute X[i+1:m-1, i] */
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i,
                            1.0f, &A[i + 1 + i * lda], lda,
                            &A[i + i * lda], lda, 0.0f, &X[i + 1 + i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, n - i, i,
                            1.0f, &Y[i], ldy, &A[i + i * lda], lda, 0.0f, &X[i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0f, &A[i + 1], lda, &X[i * ldx], 1, 1.0f, &X[i + 1 + i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, i, n - i,
                            1.0f, &A[i * lda], lda, &A[i + i * lda], lda, 0.0f, &X[i * ldx], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0f, &X[i + 1], ldx, &X[i * ldx], 1, 1.0f, &X[i + 1 + i * ldx], 1);
                cblas_sscal(m - i - 1, taup[i], &X[i + 1 + i * ldx], 1);

                /* Update A[i+1:m-1, i] */
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0f, &A[i + 1], lda, &Y[i], ldy, 1.0f, &A[i + 1 + i * lda], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            -1.0f, &X[i + 1], ldx, &A[i * lda], 1, 1.0f, &A[i + 1 + i * lda], 1);

                /* Generate reflection Q(i) to annihilate A[i+2:m-1, i] */
                slarfg(m - i - 1, &A[i + 1 + i * lda],
                       &A[((i + 2) < m ? (i + 2) : (m - 1)) + i * lda], 1, &tauq[i]);
                E[i] = A[i + 1 + i * lda];
                A[i + 1 + i * lda] = 1.0f;

                /* Compute Y[i+1:n-1, i] */
                cblas_sgemv(CblasColMajor, CblasTrans, m - i - 1, n - i - 1,
                            1.0f, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + 1 + i * lda], 1, 0.0f, &Y[i + 1 + i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, m - i - 1, i,
                            1.0f, &A[i + 1], lda, &A[i + 1 + i * lda], 1, 0.0f, &Y[i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            -1.0f, &Y[i + 1], ldy, &Y[i * ldy], 1, 1.0f, &Y[i + 1 + i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, m - i - 1, i + 1,
                            1.0f, &X[i + 1], ldx, &A[i + 1 + i * lda], 1, 0.0f, &Y[i * ldy], 1);
                cblas_sgemv(CblasColMajor, CblasTrans, i + 1, n - i - 1,
                            -1.0f, &A[(i + 1) * lda], lda, &Y[i * ldy], 1, 1.0f, &Y[i + 1 + i * ldy], 1);
                cblas_sscal(n - i - 1, tauq[i], &Y[i + 1 + i * ldy], 1);
            }
        }
    }
}
