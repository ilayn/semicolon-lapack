/**
 * @file zlahr2.c
 * @brief ZLAHR2 reduces the first NB columns of a general rectangular matrix
 *        so that elements below the k-th subdiagonal are zero.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAHR2 reduces the first NB columns of A complex general n-BY-(n-k+1)
 * matrix A so that elements below the k-th subdiagonal are zero. The
 * reduction is performed by an unitary similarity transformation
 * Q**H * A * Q. The routine returns the matrices V and T which determine
 * Q as a block reflector I - V*T*V**H, and also the matrix Y = A * V * T.
 *
 * This is an auxiliary routine called by ZGEHRD.
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
void zlahr2(const int n, const int k, const int nb,
            c128* A, const int lda, c128* tau,
            c128* T, const int ldt, c128* Y, const int ldy)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);

    int i;
    c128 ei = CMPLX(0.0, 0.0);

    /* Quick return if possible */
    if (n <= 1)
        return;

    for (i = 0; i < nb; i++) {
        if (i > 0) {
            /* Update A(k+1:n-1, i)
               Update i-th column of A - Y * V**H */
            zlacgv(i, &A[k + i], lda);
            cblas_zgemv(CblasColMajor, CblasNoTrans, n - k - 1, i, &NEG_ONE,
                        &Y[k + 1], ldy, &A[k + i], lda, &ONE,
                        &A[(k + 1) + i * lda], 1);
            zlacgv(i, &A[k + i], lda);

            /* Apply I - V * T**H * V**H to this column (call it b) from the
               left, using the last column of T as workspace

               Let  V = ( V1 )   and   b = ( b1 )   (first i rows)
                        ( V2 )             ( b2 )

               where V1 is unit lower triangular

               w := V1**H * b1 */
            cblas_zcopy(i, &A[(k + 1) + i * lda], 1, &T[(nb - 1) * ldt], 1);
            cblas_ztrmv(CblasColMajor, CblasLower, CblasConjTrans, CblasUnit,
                        i, &A[k + 1], lda, &T[(nb - 1) * ldt], 1);

            /* w := w + V2**H * b2 */
            cblas_zgemv(CblasColMajor, CblasConjTrans, n - k - i - 1, i,
                        &ONE, &A[k + i + 1], lda, &A[(k + i + 1) + i * lda], 1,
                        &ONE, &T[(nb - 1) * ldt], 1);

            /* w := T**H * w */
            cblas_ztrmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                        i, T, ldt, &T[(nb - 1) * ldt], 1);

            /* b2 := b2 - V2*w */
            cblas_zgemv(CblasColMajor, CblasNoTrans, n - k - i - 1, i, &NEG_ONE,
                        &A[k + i + 1], lda, &T[(nb - 1) * ldt], 1,
                        &ONE, &A[(k + i + 1) + i * lda], 1);

            /* b1 := b1 - V1*w */
            cblas_ztrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        i, &A[k + 1], lda, &T[(nb - 1) * ldt], 1);
            cblas_zaxpy(i, &NEG_ONE, &T[(nb - 1) * ldt], 1, &A[(k + 1) + i * lda], 1);

            A[(k + i) + (i - 1) * lda] = ei;
        }

        /* Generate the elementary reflector H(i) to annihilate A(k+i+2:n-1, i) */
        int len = n - k - i - 1;
        int start = (k + i + 2 < n) ? (k + i + 2) : (n - 1);
        zlarfg(len, &A[(k + i + 1) + i * lda], &A[start + i * lda], 1, &tau[i]);
        ei = A[(k + i + 1) + i * lda];
        A[(k + i + 1) + i * lda] = ONE;

        /* Compute Y(k+1:n-1, i) */
        cblas_zgemv(CblasColMajor, CblasNoTrans, n - k - 1, len,
                    &ONE, &A[(k + 1) + (i + 1) * lda], lda,
                    &A[(k + i + 1) + i * lda], 1,
                    &ZERO, &Y[(k + 1) + i * ldy], 1);
        cblas_zgemv(CblasColMajor, CblasConjTrans, len, i,
                    &ONE, &A[k + i + 1], lda, &A[(k + i + 1) + i * lda], 1,
                    &ZERO, &T[i * ldt], 1);
        cblas_zgemv(CblasColMajor, CblasNoTrans, n - k - 1, i, &NEG_ONE,
                    &Y[k + 1], ldy, &T[i * ldt], 1, &ONE, &Y[(k + 1) + i * ldy], 1);
        cblas_zscal(n - k - 1, &tau[i], &Y[(k + 1) + i * ldy], 1);

        /* Compute T(0:i, i) */
        {
            c128 neg_tau = -tau[i];
            cblas_zscal(i, &neg_tau, &T[i * ldt], 1);
        }
        cblas_ztrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    i, T, ldt, &T[i * ldt], 1);
        T[i + i * ldt] = tau[i];
    }
    A[(k + nb) + (nb - 1) * lda] = ei;

    /* Compute Y(0:k, 0:nb-1) */
    zlacpy("All", k + 1, nb, &A[lda], lda, Y, ldy);
    cblas_ztrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                k + 1, nb, &ONE, &A[k + 1], lda, Y, ldy);
    if (n > k + 1 + nb) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k + 1, nb,
                    n - k - 1 - nb,
                    &ONE, &A[(nb + 1) * lda], lda, &A[k + nb + 1], lda,
                    &ONE, Y, ldy);
    }
    cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                k + 1, nb, &ONE, T, ldt, Y, ldy);
}
