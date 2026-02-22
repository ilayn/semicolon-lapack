/**
 * @file dlabrd.c
 * @brief DLABRD reduces the first nb rows and columns of a general matrix to bidiagonal form.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include <cblas.h>

/**
 * DLABRD reduces the first NB rows and columns of a real general
 * m by n matrix A to upper or lower bidiagonal form by an orthogonal
 * transformation Q**T * A * P, and returns the matrices X and Y which
 * are needed to apply the transformation to the unreduced part of A.
 *
 * If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
 * bidiagonal form.
 *
 * This is an auxiliary routine called by DGEBRD.
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
void dlabrd(const INT m, const INT n, const INT nb,
            f64* restrict A, const INT lda,
            f64* restrict D, f64* restrict E,
            f64* restrict tauq, f64* restrict taup,
            f64* restrict X, const INT ldx,
            f64* restrict Y, const INT ldy)
{
    INT i;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i:m-1, i] */
            cblas_dgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        -1.0, &A[i], lda, &Y[i], ldy, 1.0, &A[i + i * lda], 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        -1.0, &X[i], ldx, &A[i * lda], 1, 1.0, &A[i + i * lda], 1);

            /* Generate reflection Q(i) to annihilate A[i+1:m-1, i] */
            dlarfg(m - i, &A[i + i * lda], &A[((i + 1) < m ? (i + 1) : (m - 1)) + i * lda], 1,
                   &tauq[i]);
            D[i] = A[i + i * lda];

            if (i < n - 1) {
                A[i + i * lda] = 1.0;

                /* Compute Y[i+1:n-1, i] */
                cblas_dgemv(CblasColMajor, CblasTrans, m - i, n - i - 1,
                            1.0, &A[i + (i + 1) * lda], lda,
                            &A[i + i * lda], 1, 0.0, &Y[i + 1 + i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, m - i, i,
                            1.0, &A[i], lda, &A[i + i * lda], 1, 0.0, &Y[i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            -1.0, &Y[i + 1], ldy, &Y[i * ldy], 1, 1.0, &Y[i + 1 + i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, m - i, i,
                            1.0, &X[i], ldx, &A[i + i * lda], 1, 0.0, &Y[i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, i, n - i - 1,
                            -1.0, &A[(i + 1) * lda], lda, &Y[i * ldy], 1, 1.0, &Y[i + 1 + i * ldy], 1);
                cblas_dscal(n - i - 1, tauq[i], &Y[i + 1 + i * ldy], 1);

                /* Update A[i, i+1:n-1] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, n - i - 1, i + 1,
                            -1.0, &Y[i + 1], ldy, &A[i], lda, 1.0, &A[i + (i + 1) * lda], lda);
                cblas_dgemv(CblasColMajor, CblasTrans, i, n - i - 1,
                            -1.0, &A[(i + 1) * lda], lda, &X[i], ldx, 1.0, &A[i + (i + 1) * lda], lda);

                /* Generate reflection P(i) to annihilate A[i, i+2:n-1] */
                dlarfg(n - i - 1, &A[i + (i + 1) * lda],
                       &A[i + ((i + 2) < n ? (i + 2) : (n - 1)) * lda], lda, &taup[i]);
                E[i] = A[i + (i + 1) * lda];
                A[i + (i + 1) * lda] = 1.0;

                /* Compute X[i+1:m-1, i] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i - 1,
                            1.0, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda, 0.0, &X[i + 1 + i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, n - i - 1, i + 1,
                            1.0, &Y[i + 1], ldy, &A[i + (i + 1) * lda], lda, 0.0, &X[i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            -1.0, &A[i + 1], lda, &X[i * ldx], 1, 1.0, &X[i + 1 + i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, i, n - i - 1,
                            1.0, &A[(i + 1) * lda], lda, &A[i + (i + 1) * lda], lda, 0.0, &X[i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0, &X[i + 1], ldx, &X[i * ldx], 1, 1.0, &X[i + 1 + i * ldx], 1);
                cblas_dscal(m - i - 1, taup[i], &X[i + 1 + i * ldx], 1);
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i, i:n-1] */
            cblas_dgemv(CblasColMajor, CblasNoTrans, n - i, i,
                        -1.0, &Y[i], ldy, &A[i], lda, 1.0, &A[i + i * lda], lda);
            cblas_dgemv(CblasColMajor, CblasTrans, i, n - i,
                        -1.0, &A[i * lda], lda, &X[i], ldx, 1.0, &A[i + i * lda], lda);

            /* Generate reflection P(i) to annihilate A[i, i+1:n-1] */
            dlarfg(n - i, &A[i + i * lda], &A[i + ((i + 1) < n ? (i + 1) : (n - 1)) * lda], lda,
                   &taup[i]);
            D[i] = A[i + i * lda];

            if (i < m - 1) {
                A[i + i * lda] = 1.0;

                /* Compute X[i+1:m-1, i] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i,
                            1.0, &A[i + 1 + i * lda], lda,
                            &A[i + i * lda], lda, 0.0, &X[i + 1 + i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, n - i, i,
                            1.0, &Y[i], ldy, &A[i + i * lda], lda, 0.0, &X[i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0, &A[i + 1], lda, &X[i * ldx], 1, 1.0, &X[i + 1 + i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, i, n - i,
                            1.0, &A[i * lda], lda, &A[i + i * lda], lda, 0.0, &X[i * ldx], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0, &X[i + 1], ldx, &X[i * ldx], 1, 1.0, &X[i + 1 + i * ldx], 1);
                cblas_dscal(m - i - 1, taup[i], &X[i + 1 + i * ldx], 1);

                /* Update A[i+1:m-1, i] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            -1.0, &A[i + 1], lda, &Y[i], ldy, 1.0, &A[i + 1 + i * lda], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            -1.0, &X[i + 1], ldx, &A[i * lda], 1, 1.0, &A[i + 1 + i * lda], 1);

                /* Generate reflection Q(i) to annihilate A[i+2:m-1, i] */
                dlarfg(m - i - 1, &A[i + 1 + i * lda],
                       &A[((i + 2) < m ? (i + 2) : (m - 1)) + i * lda], 1, &tauq[i]);
                E[i] = A[i + 1 + i * lda];
                A[i + 1 + i * lda] = 1.0;

                /* Compute Y[i+1:n-1, i] */
                cblas_dgemv(CblasColMajor, CblasTrans, m - i - 1, n - i - 1,
                            1.0, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + 1 + i * lda], 1, 0.0, &Y[i + 1 + i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, m - i - 1, i,
                            1.0, &A[i + 1], lda, &A[i + 1 + i * lda], 1, 0.0, &Y[i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            -1.0, &Y[i + 1], ldy, &Y[i * ldy], 1, 1.0, &Y[i + 1 + i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, m - i - 1, i + 1,
                            1.0, &X[i + 1], ldx, &A[i + 1 + i * lda], 1, 0.0, &Y[i * ldy], 1);
                cblas_dgemv(CblasColMajor, CblasTrans, i + 1, n - i - 1,
                            -1.0, &A[(i + 1) * lda], lda, &Y[i * ldy], 1, 1.0, &Y[i + 1 + i * ldy], 1);
                cblas_dscal(n - i - 1, tauq[i], &Y[i + 1 + i * ldy], 1);
            }
        }
    }
}
