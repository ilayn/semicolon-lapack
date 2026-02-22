/**
 * @file dorm2l.c
 * @brief DORM2L multiplies a general matrix by the orthogonal matrix from
 *        a QL factorization determined by DGEQLF (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DORM2L overwrites the general real m by n matrix C with
 *
 *    Q * C   if SIDE = 'L' and TRANS = "N", or
 *    Q^T * C if SIDE = 'L' and TRANS = "T", or
 *    C * Q   if SIDE = 'R' and TRANS = "N", or
 *    C * Q^T if SIDE = 'R' and TRANS = "T",
 *
 * where Q is a real orthogonal matrix defined as the product of k
 * elementary reflectors
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by DGEQLF. Q is of order m if SIDE = 'L' and of order n
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
 * @param[in]     A      The i-th column must contain the vector which defines
 *                       the elementary reflector H(i), as returned by DGEQLF
 *                       in the last k columns. Dimension (lda, k).
 * @param[in]     lda    Leading dimension of A.
 *                       If SIDE = "L", lda >= max(1, m);
 *                       if SIDE = "R", lda >= max(1, n).
 * @param[in]     tau    Array of dimension (k). TAU(i) is the scalar factor
 *                       of H(i), as returned by DGEQLF.
 * @param[in,out] C      On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if SIDE = "L",
 *                                   dimension (m) if SIDE = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dorm2l(const char* side, const char* trans,
            const INT m, const INT n, const INT k,
            const f64* restrict A, const INT lda,
            const f64* restrict tau,
            f64* restrict C, const INT ldc,
            f64* restrict work,
            INT* info)
{
    INT i, mi, ni, nq;
    INT left, notran;
    INT i1, i2, i3;

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
    } else if (lda < (nq > 1 ? nq : 1)) {
        *info = -7;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DORM2L", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    /* Determine loop direction:
     * Left+NoTrans or Right+Trans: i = 0, 1, ..., k-1 (forward)
     * Left+Trans or Right+NoTrans: i = k-1, k-2, ..., 0 (backward) */
    if ((left && notran) || (!left && !notran)) {
        i1 = 0;
        i2 = k - 1;
        i3 = 1;
    } else {
        i1 = k - 1;
        i2 = 0;
        i3 = -1;
    }

    if (left) {
        ni = n;
    } else {
        mi = m;
    }

    for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
        if (left) {
            /* H(i) is applied to C(0:m-k+i, 0:n-1) */
            mi = m - k + i + 1;
        } else {
            /* H(i) is applied to C(0:m-1, 0:n-k+i) */
            ni = n - k + i + 1;
        }

        /* Apply H(i) - note: reflector stored in column i of A,
         * with v(last) = 1, so we use dlarf1l */
        dlarf1l(side, mi, ni, &A[0 + i * lda], 1, tau[i],
                C, ldc, work);
    }
}
