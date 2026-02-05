/**
 * @file dlahr2.c
 * @brief DLAHR2 reduces the first NB columns of a general rectangular matrix
 *        so that elements below the k-th subdiagonal are zero.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLAHR2 reduces the first NB columns of A real general n-BY-(n-k+1)
 * matrix A so that elements below the k-th subdiagonal are zero. The
 * reduction is performed by an orthogonal similarity transformation
 * Q**T * A * Q. The routine returns the matrices V and T which determine
 * Q as a block reflector I - V*T*V**T, and also the matrix Y = A * V * T.
 *
 * This is an auxiliary routine called by DGEHRD.
 *
 * @param[in]     n      The order of the matrix A.
 * @param[in]     k      The offset for the reduction. Elements below the k-th
 *                       subdiagonal in the first NB columns are reduced to zero.
 *                       k < n. (0-based: corresponds to Fortran K-1)
 * @param[in]     nb     The number of columns to be reduced.
 * @param[in,out] A      On entry, the n-by-(n-k) general matrix A.
 *                       On exit, the elements on and above the k-th subdiagonal
 *                       in the first NB columns are overwritten with the
 *                       corresponding elements of the reduced matrix.
 *                       Dimension (lda, n-k).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[out]    tau    The scalar factors of the elementary reflectors.
 *                       Dimension (nb).
 * @param[out]    T      The upper triangular matrix T. Dimension (ldt, nb).
 * @param[in]     ldt    The leading dimension of T. ldt >= nb.
 * @param[out]    Y      The n-by-nb matrix Y. Dimension (ldy, nb).
 * @param[in]     ldy    The leading dimension of Y. ldy >= n.
 */
void dlahr2(const int n, const int k, const int nb,
            double* A, const int lda, double* tau,
            double* T, const int ldt, double* Y, const int ldy)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i;
    double ei = 0.0;

    /* Quick return if possible */
    if (n <= 1)
        return;

    for (i = 0; i < nb; i++) {
        if (i > 0) {
            /* Update A(k:n-1, i)
               Update i-th column of A - Y * V**T */
            cblas_dgemv(CblasColMajor, CblasNoTrans, n - k, i, -ONE,
                        &Y[k], ldy, &A[k + i - 1], lda, ONE, &A[k + i * lda], 1);

            /* Apply I - V * T**T * V**T to this column (call it b) from the
               left, using the last column of T as workspace

               Let  V = ( V1 )   and   b = ( b1 )   (first i rows)
                        ( V2 )             ( b2 )

               where V1 is unit lower triangular

               w := V1**T * b1 */
            cblas_dcopy(i, &A[k + i * lda], 1, &T[(nb - 1) * ldt], 1);
            cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasUnit,
                        i, &A[k], lda, &T[(nb - 1) * ldt], 1);

            /* w := w + V2**T * b2 */
            cblas_dgemv(CblasColMajor, CblasTrans, n - k - i, i,
                        ONE, &A[k + i], lda, &A[k + i + i * lda], 1,
                        ONE, &T[(nb - 1) * ldt], 1);

            /* w := T**T * w */
            cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        i, T, ldt, &T[(nb - 1) * ldt], 1);

            /* b2 := b2 - V2*w */
            cblas_dgemv(CblasColMajor, CblasNoTrans, n - k - i, i, -ONE,
                        &A[k + i], lda, &T[(nb - 1) * ldt], 1,
                        ONE, &A[k + i + i * lda], 1);

            /* b1 := b1 - V1*w */
            cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        i, &A[k], lda, &T[(nb - 1) * ldt], 1);
            cblas_daxpy(i, -ONE, &T[(nb - 1) * ldt], 1, &A[k + i * lda], 1);

            A[k + i - 1 + (i - 1) * lda] = ei;
        }

        /* Generate the elementary reflector H(i) to annihilate A(k+i+1:n-1, i) */
        int len = n - k - i;
        int start = (k + i + 1 < n) ? (k + i + 1) : (n - 1);
        dlarfg(len, &A[k + i + i * lda], &A[start + i * lda], 1, &tau[i]);
        ei = A[k + i + i * lda];
        A[k + i + i * lda] = ONE;

        /* Compute Y(k:n-1, i) */
        cblas_dgemv(CblasColMajor, CblasNoTrans, n - k, len,
                    ONE, &A[k + (i + 1) * lda], lda, &A[k + i + i * lda], 1,
                    ZERO, &Y[k + i * ldy], 1);
        cblas_dgemv(CblasColMajor, CblasTrans, len, i,
                    ONE, &A[k + i], lda, &A[k + i + i * lda], 1,
                    ZERO, &T[i * ldt], 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, n - k, i, -ONE,
                    &Y[k], ldy, &T[i * ldt], 1, ONE, &Y[k + i * ldy], 1);
        cblas_dscal(n - k, tau[i], &Y[k + i * ldy], 1);

        /* Compute T(0:i, i) */
        cblas_dscal(i, -tau[i], &T[i * ldt], 1);
        cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    i, T, ldt, &T[i * ldt], 1);
        T[i + i * ldt] = tau[i];
    }
    A[k + nb - 1 + (nb - 1) * lda] = ei;

    /* Compute Y(0:k, 0:nb-1) */
    dlacpy("All", k, nb, &A[lda], lda, Y, ldy);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                k, nb, ONE, &A[k], lda, Y, ldy);
    if (n > k + nb) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, nb, n - k - nb,
                    ONE, &A[(nb + 1) * lda], lda, &A[k + nb], lda, ONE, Y, ldy);
    }
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                k, nb, ONE, T, ldt, Y, ldy);
}
