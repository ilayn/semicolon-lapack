/**
 * @file cgelqt.c
 * @brief CGELQT computes a blocked LQ factorization of a general M-by-N matrix
 *        using the compact WY representation of Q.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CGELQT computes a blocked LQ factorization of a complex M-by-N matrix A
 * using the compact WY representation of Q.
 *
 * The matrix V stores the elementary reflectors H(i) in the i-th row
 * above the diagonal. For example, if M=3 and N=5, the matrix V is
 *
 *              V = (  1  v1 v1 v1 v1 )
 *                  (     1  v2 v2 v2 )
 *                  (         1 v3 v3 )
 *
 * where the vi's represent the vectors which define H(i), which are returned
 * in the matrix A. The 1's along the diagonal of V are not stored in A.
 *
 * Let K=MIN(M,N). The number of blocks is B = ceiling(K/MB), where each
 * block is of order MB except for the last block, which is of order
 * IB = K - (B-1)*MB. For each of the B blocks, an upper triangular block
 * reflector factor is computed: T1, T2, ..., TB. The MB-by-MB (and IB-by-IB
 * for the last block) T's are stored in the MB-by-K matrix T as
 *
 *              T = (T1 T2 ... TB).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     mb    The block size to be used in the blocked LQ.
 *                      min(m,n) >= mb >= 1.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the M-by-N matrix A.
 *                      On exit, the elements on and below the diagonal of the
 *                      array contain the M-by-min(M,N) lower trapezoidal matrix L
 *                      (L is lower triangular if M <= N); the elements above the
 *                      diagonal are the rows of V.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T     Single complex array, dimension (ldt, min(m,n)).
 *                      The upper triangular block reflectors stored in compact form
 *                      as a sequence of upper triangular blocks.
 * @param[in]     ldt   The leading dimension of the array T. ldt >= mb.
 * @param[out]    work  Single complex array, dimension (mb*m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cgelqt(const INT m, const INT n, const INT mb,
            c64* restrict A, const INT lda,
            c64* restrict T, const INT ldt,
            c64* restrict work,
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
    } else if (mb < 1 || (mb > minmn && minmn > 0)) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldt < mb) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("CGELQT", -(*info));
        return;
    }

    /* Quick return if possible */
    k = minmn;
    if (k == 0) {
        return;
    }

    /* Blocked loop of length K */
    for (i = 0; i < k; i += mb) {
        ib = (k - i) < mb ? (k - i) : mb;

        /* Compute the LQ factorization of the current block A(i:i+ib-1, i:n-1) */
        cgelqt3(ib, n - i, &A[i + i * lda], lda, &T[0 + i * ldt], ldt, &iinfo);

        if (i + ib < m) {
            /* Update by applying H^T to A(i+ib:m-1, i:n-1) from the right */
            clarfb("R", "N", "F", "R",
                   m - i - ib, n - i, ib,
                   &A[i + i * lda], lda,
                   &T[0 + i * ldt], ldt,
                   &A[(i + ib) + i * lda], lda,
                   work, m - i - ib);
        }
    }
}
