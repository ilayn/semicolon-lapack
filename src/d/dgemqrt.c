/**
 * @file dgemqrt.c
 * @brief DGEMQRT overwrites the general real M-by-N matrix C with the product
 *        of Q (or Q^T) and C, using the compact WY representation from DGEQRT.
 */

#include "semicolon_lapack_double.h"

/**
 * DGEMQRT overwrites the general real M-by-N matrix C with
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
 * generated using the compact WY representation as returned by DGEQRT.
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
 * @param[in]     nb     The block size used for the storage of T. k >= nb >= 1.
 *                       This must be the same value of nb used to generate T
 *                       in DGEQRT.
 * @param[in]     V      Double precision array, dimension (ldv, k).
 *                       The i-th column must contain the vector which defines
 *                       the elementary reflector H(i), for i = 0,1,...,k-1,
 *                       as returned by DGEQRT in the first k columns of its
 *                       array argument A.
 * @param[in]     ldv    The leading dimension of the array V.
 *                       If SIDE = "L", ldv >= max(1, m);
 *                       if SIDE = "R", ldv >= max(1, n).
 * @param[in]     T      Double precision array, dimension (ldt, k).
 *                       The upper triangular factors of the block reflectors
 *                       as returned by DGEQRT, stored as an nb-by-k matrix.
 * @param[in]     ldt    The leading dimension of the array T. ldt >= nb.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C, Q^T*C, C*Q^T, or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Double precision workspace array.
 *                       Dimension is n*nb if SIDE = "L", or m*nb if SIDE = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgemqrt(const char* side, const char* trans,
             const int m, const int n, const int k, const int nb,
             const f64* restrict V, const int ldv,
             const f64* restrict T, const int ldt,
             f64* restrict C, const int ldc,
             f64* restrict work, int* info)
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
    } else if (nb < 1 || (nb > k && k > 0)) {
        *info = -6;
    } else if (ldv < (q > 1 ? q : 1)) {
        *info = -8;
    } else if (ldt < nb) {
        *info = -10;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        xerbla("DGEMQRT", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    if (left && tran) {
        /* Case: Q^T * C -- forward loop */
        for (i = 0; i < k; i += nb) {
            ib = nb < k - i ? nb : k - i;
            dlarfb("L", "T", "F", "C", m - i, n, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[i + 0 * ldc], ldc,
                   work, ldwork);
        }

    } else if (right && notran) {
        /* Case: C * Q -- forward loop */
        for (i = 0; i < k; i += nb) {
            ib = nb < k - i ? nb : k - i;
            dlarfb("R", "N", "F", "C", m, n - i, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[0 + i * ldc], ldc,
                   work, ldwork);
        }

    } else if (left && notran) {
        /* Case: Q * C -- backward loop */
        kf = ((k - 1) / nb) * nb;
        for (i = kf; i >= 0; i -= nb) {
            ib = nb < k - i ? nb : k - i;
            dlarfb("L", "N", "F", "C", m - i, n, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[i + 0 * ldc], ldc,
                   work, ldwork);
        }

    } else if (right && tran) {
        /* Case: C * Q^T -- backward loop */
        kf = ((k - 1) / nb) * nb;
        for (i = kf; i >= 0; i -= nb) {
            ib = nb < k - i ? nb : k - i;
            dlarfb("R", "T", "F", "C", m, n - i, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[0 + i * ldc], ldc,
                   work, ldwork);
        }
    }
}
