/**
 * @file sgemlqt.c
 * @brief SGEMLQT overwrites the general real M-by-N matrix C with the product
 *        of Q (or Q^T) and C, using the compact WY representation from SGELQT.
 */

#include "semicolon_lapack_single.h"

/**
 * SGEMLQT overwrites the general real M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'T':   Q^T * C        C * Q^T
 *
 * where Q is a real orthogonal matrix defined as the product of K
 * elementary reflectors:
 *
 *    Q = H(0) H(1) . . . H(K-1) = I - V T V^T
 *
 * generated using the compact WY representation as returned by SGELQT.
 *
 * Q is of order M if SIDE = 'L' and of order N if SIDE = 'R'.
 *
 * @param[in]     side   'L': apply Q or Q^T from the Left;
 *                       'R': apply Q or Q^T from the Right.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'T': Transpose, apply Q^T.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     mb     The block size used for the storage of T. k >= mb >= 1.
 *                       This must be the same value of mb used to generate T
 *                       in SGELQT.
 * @param[in]     V      Double precision array, dimension (ldv, m) if SIDE = 'L',
 *                       (ldv, n) if SIDE = 'R'.
 *                       The i-th row must contain the vector which defines
 *                       the elementary reflector H(i), for i = 0,1,...,k-1,
 *                       as returned by SGELQT in the first k rows of its
 *                       array argument A.
 * @param[in]     ldv    The leading dimension of the array V. ldv >= max(1, k).
 * @param[in]     T      Double precision array, dimension (ldt, k).
 *                       The upper triangular factors of the block reflectors
 *                       as returned by SGELQT, stored as an mb-by-k matrix.
 * @param[in]     ldt    The leading dimension of the array T. ldt >= mb.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C, Q^T*C, C*Q^T, or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Double precision workspace array.
 *                       Dimension is n*mb if SIDE = "L", or m*mb if SIDE = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgemlqt(const char* side, const char* trans,
             const int m, const int n, const int k, const int mb,
             const f32* const restrict V, const int ldv,
             const f32* const restrict T, const int ldt,
             f32* const restrict C, const int ldc,
             f32* const restrict work, int* info)
{
    int left, right, tran, notran;
    int i, ib, ldwork, kf, q;

    /* Decode arguments */
    left   = (side[0] == 'L' || side[0] == 'l');
    right  = (side[0] == 'R' || side[0] == 'r');
    tran   = (trans[0] == 'T' || trans[0] == 't');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    *info = 0;
    if (left) {
        ldwork = n > 1 ? n : 1;
        q = m;
    } else if (right) {
        ldwork = m > 1 ? m : 1;
        q = n;
    }

    /* Parameter validation */
    if (!left && !right) {
        *info = -1;
    } else if (!tran && !notran) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > q) {
        *info = -5;
    } else if (mb < 1 || (mb > k && k > 0)) {
        *info = -6;
    } else if (ldv < (k > 1 ? k : 1)) {
        *info = -8;
    } else if (ldt < mb) {
        *info = -10;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        xerbla("SGEMLQT", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    if (left && notran) {
        /* Case: Q * C -- forward loop */
        for (i = 0; i < k; i += mb) {
            ib = mb < k - i ? mb : k - i;
            slarfb("L", "T", "F", "R", m - i, n, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[i + 0 * ldc], ldc,
                   work, ldwork);
        }

    } else if (right && tran) {
        /* Case: C * Q^T -- forward loop */
        for (i = 0; i < k; i += mb) {
            ib = mb < k - i ? mb : k - i;
            slarfb("R", "N", "F", "R", m, n - i, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[0 + i * ldc], ldc,
                   work, ldwork);
        }

    } else if (left && tran) {
        /* Case: Q^T * C -- backward loop */
        kf = ((k - 1) / mb) * mb;
        for (i = kf; i >= 0; i -= mb) {
            ib = mb < k - i ? mb : k - i;
            slarfb("L", "N", "F", "R", m - i, n, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[i + 0 * ldc], ldc,
                   work, ldwork);
        }

    } else if (right && notran) {
        /* Case: C * Q -- backward loop */
        kf = ((k - 1) / mb) * mb;
        for (i = kf; i >= 0; i -= mb) {
            ib = mb < k - i ? mb : k - i;
            slarfb("R", "T", "F", "R", m, n - i, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[0 + i * ldc], ldc,
                   work, ldwork);
        }
    }
}
