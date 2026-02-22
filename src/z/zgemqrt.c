/**
 * @file zgemqrt.c
 * @brief ZGEMQRT overwrites the general complex M-by-N matrix C with the product
 *        of Q (or Q^H) and C, using the compact WY representation from ZGEQRT.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZGEMQRT overwrites the general complex M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'C':   Q^H * C        C * Q^H
 *
 * where Q is a complex unitary matrix defined as the product of K
 * elementary reflectors:
 *
 *    Q = H(0) H(1) . . . H(K-1) = I - V T V^H
 *
 * generated using the compact WY representation as returned by ZGEQRT.
 *
 * Q is of order M if SIDE = 'L' and of order N if SIDE = 'R'.
 *
 * @param[in]     side   'L': apply Q or Q^H from the Left;
 *                       'R': apply Q or Q^H from the Right.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'C': Conjugate transpose, apply Q^H.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q.
 *                       If SIDE = "L", m >= k >= 0;
 *                       if SIDE = "R", n >= k >= 0.
 * @param[in]     nb     The block size used for the storage of T. k >= nb >= 1.
 *                       This must be the same value of nb used to generate T
 *                       in ZGEQRT.
 * @param[in]     V      Double complex array, dimension (ldv, k).
 *                       The i-th column must contain the vector which defines
 *                       the elementary reflector H(i), for i = 0,1,...,k-1,
 *                       as returned by ZGEQRT in the first k columns of its
 *                       array argument A.
 * @param[in]     ldv    The leading dimension of the array V.
 *                       If SIDE = "L", ldv >= max(1, m);
 *                       if SIDE = "R", ldv >= max(1, n).
 * @param[in]     T      Double complex array, dimension (ldt, k).
 *                       The upper triangular factors of the block reflectors
 *                       as returned by ZGEQRT, stored as an nb-by-k matrix.
 * @param[in]     ldt    The leading dimension of the array T. ldt >= nb.
 * @param[in,out] C      Double complex array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C, Q^H*C, C*Q^H, or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Double complex workspace array.
 *                       Dimension is n*nb if SIDE = "L", or m*nb if SIDE = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zgemqrt(const char* side, const char* trans,
             const INT m, const INT n, const INT k, const INT nb,
             const c128* restrict V, const INT ldv,
             const c128* restrict T, const INT ldt,
             c128* restrict C, const INT ldc,
             c128* restrict work, INT* info)
{
    INT left, right, tran, notran;
    INT i, ib, ldwork, kf, q;

    /* Decode arguments */
    left   = (side[0] == 'L' || side[0] == 'l');
    right  = (side[0] == 'R' || side[0] == 'r');
    tran   = (trans[0] == 'C' || trans[0] == 'c');
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
        xerbla("ZGEMQRT", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    if (left && tran) {
        /* Case: Q^H * C -- forward loop */
        for (i = 0; i < k; i += nb) {
            ib = nb < k - i ? nb : k - i;
            zlarfb("L", "C", "F", "C", m - i, n, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[i + 0 * ldc], ldc,
                   work, ldwork);
        }

    } else if (right && notran) {
        /* Case: C * Q -- forward loop */
        for (i = 0; i < k; i += nb) {
            ib = nb < k - i ? nb : k - i;
            zlarfb("R", "N", "F", "C", m, n - i, ib,
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
            zlarfb("L", "N", "F", "C", m - i, n, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[i + 0 * ldc], ldc,
                   work, ldwork);
        }

    } else if (right && tran) {
        /* Case: C * Q^H -- backward loop */
        kf = ((k - 1) / nb) * nb;
        for (i = kf; i >= 0; i -= nb) {
            ib = nb < k - i ? nb : k - i;
            zlarfb("R", "C", "F", "C", m, n - i, ib,
                   &V[i + i * ldv], ldv,
                   &T[0 + i * ldt], ldt,
                   &C[0 + i * ldc], ldc,
                   work, ldwork);
        }
    }
}
