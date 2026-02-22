/**
 * @file dlahr2.c
 * @brief DLAHR2 reduces the first NB columns of a general rectangular matrix
 *        so that elements below the k-th subdiagonal are zero.
 */

#include "internal_build_defs.h"
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
 *                       k < n. (0-based)
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
void dlahr2(const INT n, const INT k, const INT nb,
            f64* A, const INT lda, f64* tau,
            f64* T, const INT ldt, f64* Y, const INT ldy)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i;
    f64 ei = 0.0;

    /* Quick return if possible */
    if (n <= 1)
        return;

    for (i = 0; i < nb; i++) {
        if (i > 0) {
            /* Update A(k+1:n-1, i)
               Update i-th column of A - Y * V**T */
            cblas_dgemv(CblasColMajor, CblasNoTrans, n - k - 1, i, -ONE,
                        &Y[k + 1], ldy, &A[k + i], lda, ONE,
                        &A[(k + 1) + i * lda], 1);

            /* Apply I - V * T**T * V**T to this column (call it b) from the
               left, using the last column of T as workspace

               Let  V = ( V1 )   and   b = ( b1 )   (first i rows)
                        ( V2 )             ( b2 )

               where V1 is unit lower triangular

               w := V1**T * b1 */
            cblas_dcopy(i, &A[(k + 1) + i * lda], 1, &T[(nb - 1) * ldt], 1);
            cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasUnit,
                        i, &A[k + 1], lda, &T[(nb - 1) * ldt], 1);

            /* w := w + V2**T * b2 */
            cblas_dgemv(CblasColMajor, CblasTrans, n - k - i - 1, i,
                        ONE, &A[k + i + 1], lda, &A[(k + i + 1) + i * lda], 1,
                        ONE, &T[(nb - 1) * ldt], 1);

            /* w := T**T * w */
            cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        i, T, ldt, &T[(nb - 1) * ldt], 1);

            /* b2 := b2 - V2*w */
            cblas_dgemv(CblasColMajor, CblasNoTrans, n - k - i - 1, i, -ONE,
                        &A[k + i + 1], lda, &T[(nb - 1) * ldt], 1,
                        ONE, &A[(k + i + 1) + i * lda], 1);

            /* b1 := b1 - V1*w */
            cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        i, &A[k + 1], lda, &T[(nb - 1) * ldt], 1);
            cblas_daxpy(i, -ONE, &T[(nb - 1) * ldt], 1, &A[(k + 1) + i * lda], 1);

            A[(k + i) + (i - 1) * lda] = ei;
        }

        /* Generate the elementary reflector H(i) to annihilate A(k+i+2:n-1, i) */
        INT len = n - k - i - 1;
        INT start = (k + i + 2 < n) ? (k + i + 2) : (n - 1);
        dlarfg(len, &A[(k + i + 1) + i * lda], &A[start + i * lda], 1, &tau[i]);
        ei = A[(k + i + 1) + i * lda];
        A[(k + i + 1) + i * lda] = ONE;

        /* Compute Y(k+1:n-1, i) */
        cblas_dgemv(CblasColMajor, CblasNoTrans, n - k - 1, len,
                    ONE, &A[(k + 1) + (i + 1) * lda], lda,
                    &A[(k + i + 1) + i * lda], 1,
                    ZERO, &Y[(k + 1) + i * ldy], 1);
        cblas_dgemv(CblasColMajor, CblasTrans, len, i,
                    ONE, &A[k + i + 1], lda, &A[(k + i + 1) + i * lda], 1,
                    ZERO, &T[i * ldt], 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, n - k - 1, i, -ONE,
                    &Y[k + 1], ldy, &T[i * ldt], 1, ONE, &Y[(k + 1) + i * ldy], 1);
        cblas_dscal(n - k - 1, tau[i], &Y[(k + 1) + i * ldy], 1);

        /* Compute T(0:i, i) */
        cblas_dscal(i, -tau[i], &T[i * ldt], 1);
        cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    i, T, ldt, &T[i * ldt], 1);
        T[i + i * ldt] = tau[i];
    }
    A[(k + nb) + (nb - 1) * lda] = ei;

    /* Compute Y(0:k, 0:nb-1) */
    dlacpy("All", k + 1, nb, &A[lda], lda, Y, ldy);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                k + 1, nb, ONE, &A[k + 1], lda, Y, ldy);
    if (n > k + 1 + nb) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k + 1, nb,
                    n - k - 1 - nb,
                    ONE, &A[(nb + 1) * lda], lda, &A[k + nb + 1], lda,
                    ONE, Y, ldy);
    }
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                k + 1, nb, ONE, T, ldt, Y, ldy);
}
