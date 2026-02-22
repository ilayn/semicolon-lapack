#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
/**
 * @file sgeqrt.c
 * @brief SGEQRT computes a blocked QR factorization of a general M-by-N matrix
 *        using the compact WY representation of Q.
 */

/**
 * SGEQRT computes a blocked QR factorization of a real M-by-N matrix A
 * using the compact WY representation of Q.
 *
 * The factorization has the form
 *    A = Q * R
 * where Q is represented in the compact WY form as a product of elementary
 * reflectors stored with their triangular block reflector factors T.
 *
 * The matrix V stores the elementary reflectors H(i) in the i-th column
 * below the diagonal. For example, if M=5 and N=3, the matrix V is
 *
 *              V = (  1       )
 *                  ( v1  1    )
 *                  ( v1 v2  1 )
 *                  ( v1 v2 v3 )
 *                  ( v1 v2 v3 )
 *
 * where the vi's represent the vectors which define H(i), which are returned
 * in the matrix A. The 1's along the diagonal of V are not stored in A.
 *
 * Let K=MIN(M,N). The number of blocks is B = ceiling(K/NB), where each
 * block is of order NB except for the last block, which is of order
 * IB = K - (B-1)*NB. For each of the B blocks, an upper triangular block
 * reflector factor is computed: T1, T2, ..., TB. The NB-by-NB (and IB-by-IB
 * for the last block) T's are stored in the NB-by-K matrix T as
 *
 *              T = (T1 T2 ... TB).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     nb    The block size to be used in the blocked QR.
 *                      min(m,n) >= nb >= 1.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the M-by-N matrix A.
 *                      On exit, the elements on and above the diagonal of the
 *                      array contain the min(M,N)-by-N upper trapezoidal matrix R
 *                      (R is upper triangular if M >= N); the elements below the
 *                      diagonal are the columns of V.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T     Double precision array, dimension (ldt, min(m,n)).
 *                      The upper triangular block reflectors stored in compact form
 *                      as a sequence of upper triangular blocks.
 * @param[in]     ldt   The leading dimension of the array T. ldt >= nb.
 * @param[out]    work  Double precision array, dimension (nb*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgeqrt(const INT m, const INT n, const INT nb,
            f32* restrict A, const INT lda,
            f32* restrict T, const INT ldt,
            f32* restrict work,
            INT* info)
{
    INT k, i, ib, iinfo;
    INT minmn;

    /* Parameter validation */
    *info = 0;
    minmn = m < n ? m : n;

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nb < 1 || (nb > minmn && minmn > 0)) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldt < nb) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("SGEQRT", -(*info));
        return;
    }

    /* Quick return if possible */
    k = minmn;
    if (k == 0) {
        return;
    }

    /* Blocked loop of length K */
    for (i = 0; i < k; i += nb) {
        ib = (k - i) < nb ? (k - i) : nb;

        /* Compute the QR factorization of the current block A(i:m-1, i:i+ib-1) */
        sgeqrt3(m - i, ib, &A[i + i * lda], lda, &T[0 + i * ldt], ldt, &iinfo);

        if (i + ib < n) {
            /* Update by applying H^T to A(i:m-1, i+ib:n-1) from the left */
            slarfb("L", "T", "F", "C",
                   m - i, n - i - ib, ib,
                   &A[i + i * lda], lda,
                   &T[0 + i * ldt], ldt,
                   &A[i + (i + ib) * lda], lda,
                   work, n - i - ib);
        }
    }
}
