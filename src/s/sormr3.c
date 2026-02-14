/**
 * @file sormr3.c
 * @brief SORMR3 multiplies a general matrix by the orthogonal matrix from
 *        a RZ factorization determined by STZRZF (unblocked algorithm).
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SORMR3 overwrites the general real m by n matrix C with
 *
 *    Q * C   if SIDE = 'L' and TRANS = "N", or
 *    Q^T * C if SIDE = 'L' and TRANS = "T", or
 *    C * Q   if SIDE = 'R' and TRANS = "N", or
 *    C * Q^T if SIDE = 'R' and TRANS = "T",
 *
 * where Q is a real orthogonal matrix defined as the product of k
 * elementary reflectors
 *
 *    Q = H(0) H(1) . . . H(k-1)
 *
 * as returned by STZRZF. Q is of order m if SIDE = 'L' and of order n
 * if SIDE = 'R'.
 *
 * @param[in]     side   'L': apply Q or Q^T from the Left;
 *                       'R': apply Q or Q^T from the Right.
 * @param[in]     trans  'N': apply Q (No transpose);
 *                       'T': apply Q^T (Transpose).
 * @param[in]     m      The number of rows of C. m >= 0.
 * @param[in]     n      The number of columns of C. n >= 0.
 * @param[in]     k      The number of elementary reflectors.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     l      The number of columns of the matrix A containing
 *                       the meaningful part of the Householder reflectors.
 *                       If SIDE = "L", m >= l >= 0;
 *                       if SIDE = "R", n >= l >= 0.
 * @param[in]     A      The i-th row must contain the vector which defines
 *                       the elementary reflector H(i), as returned by STZRZF
 *                       in the last k rows. Dimension (lda, m) if SIDE = "L",
 *                       (lda, n) if SIDE = 'R'. A is modified by the routine
 *                       but restored on exit.
 * @param[in]     lda    Leading dimension of A. lda >= max(1, k).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by STZRZF.
 * @param[in,out] C      On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if SIDE = "L",
 *                                   dimension (m) if SIDE = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sormr3(const char* side, const char* trans,
            const int m, const int n, const int k, const int l,
            const f32 * const restrict A, const int lda,
            const f32 * const restrict tau,
            f32 * const restrict C, const int ldc,
            f32 * const restrict work,
            int *info)
{
    int mi, ni, nq;
    int left, notran;

    /* Decode arguments */
    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    /* NQ is the order of Q */
    nq = left ? m : n;

    /* Parameter validation */
    *info = 0;
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (l < 0 || (left && l > m) || (!left && l > n)) {
        *info = -6;
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -8;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("SORMR3", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    /* Determine loop direction:
     * (Left && Trans) or (Right && NoTrans): i = 0, 1, ..., k-1 (forward)
     * (Left && NoTrans) or (Right && Trans): i = k-1, k-2, ..., 0 (backward) */
    int i_start, i_end, i_step;
    if ((left && !notran) || (!left && notran)) {
        i_start = 0;
        i_end = k;
        i_step = 1;
    } else {
        i_start = k - 1;
        i_end = -1;
        i_step = -1;
    }

    if (left) {
        ni = n;
        /* ja = m - l: 0-based column index into A for the reflector vectors */
    } else {
        mi = m;
        /* ja = n - l: 0-based column index into A for the reflector vectors */
    }

    for (int i = i_start; i != i_end; i += i_step) {
        if (left) {
            /* H(i) or H(i)^T is applied to C(i:m-1, 0:n-1) */
            mi = m - i;
            /* slarz("L", mi, ni, l, &A[i + (m-l)*lda], lda, tau[i],
             *        &C[i + 0*ldc], ldc, work) */
            slarz("L", mi, ni, l,
                  &A[i + (m - l) * lda], lda,
                  tau[i],
                  &C[i + 0 * ldc], ldc,
                  work);
        } else {
            /* H(i) or H(i)^T is applied to C(0:m-1, i:n-1) */
            ni = n - i;
            /* slarz("R", mi, ni, l, &A[i + (n-l)*lda], lda, tau[i],
             *        &C[0 + i*ldc], ldc, work) */
            slarz("R", mi, ni, l,
                  &A[i + (n - l) * lda], lda,
                  tau[i],
                  &C[0 + i * ldc], ldc,
                  work);
        }
    }
}
