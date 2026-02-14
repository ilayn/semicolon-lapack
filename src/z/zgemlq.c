/**
 * @file zgemlq.c
 * @brief ZGEMLQ overwrites matrix C with Q*C, Q^H*C, C*Q, or C*Q^H
 *        where Q is defined by an LQ factorization from ZGELQ.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZGEMLQ overwrites the general complex M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'C':   Q^H * C        C * Q^H
 *
 * where Q is a complex unitary matrix defined as the product of blocked
 * elementary reflectors computed by short wide LQ factorization (ZGELQ)
 *
 * @param[in]     side   'L': apply Q or Q^H from the Left;
 *                       'R': apply Q or Q^H from the Right.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'C': Conjugate transpose, apply Q^H.
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q.
 *                       If SIDE = 'L', m >= k >= 0;
 *                       if SIDE = 'R', n >= k >= 0.
 * @param[in]     A      Complex*16 array, dimension (lda, m) if SIDE='L',
 *                       (lda, n) if SIDE='R'.
 *                       Part of the data structure to represent Q as returned by ZGELQ.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, k).
 * @param[in]     T      Complex*16 array, dimension (max(5, tsize)).
 *                       Part of the data structure to represent Q as returned by ZGELQ.
 * @param[in]     tsize  The dimension of the array T. tsize >= 5.
 * @param[in,out] C      Complex*16 array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C, Q^H*C, C*Q^H, or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Complex*16 workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the minimal lwork.
 * @param[in]     lwork  The dimension of the array work. lwork >= 1.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zgemlq(const char* side, const char* trans,
            const int m, const int n, const int k,
            const double complex* const restrict A, const int lda,
            const double complex* const restrict T, const int tsize,
            double complex* const restrict C, const int ldc,
            double complex* const restrict work, const int lwork,
            int* info)
{
    int left, right, tran, notran, lquery;
    int mb, nb, lw, mn, minmnk, lwmin;

    /* Decode arguments */
    lquery = (lwork == -1);
    notran = (trans[0] == 'N' || trans[0] == 'n');
    tran   = (trans[0] == 'C' || trans[0] == 'c');
    left   = (side[0] == 'L' || side[0] == 'l');
    right  = (side[0] == 'R' || side[0] == 'r');

    /* Read block sizes from T array (stored by ZGELQ) */
    mb = (int)creal(T[1]);
    nb = (int)creal(T[2]);

    if (left) {
        lw = n * mb;
        mn = m;
    } else {
        lw = m * mb;
        mn = n;
    }

    /* Compute minimum workspace */
    minmnk = m < n ? m : n;
    minmnk = minmnk < k ? minmnk : k;
    if (minmnk == 0) {
        lwmin = 1;
    } else {
        lwmin = lw > 1 ? lw : 1;
    }

    /* NBLCKS computation: present in Fortran but result unused */
    (void)((nb > k) && (mn > k));

    /* Parameter validation */
    *info = 0;
    if (!left && !right) {
        *info = -1;
    } else if (!tran && !notran) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > mn) {
        *info = -5;
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -7;
    } else if (tsize < 5) {
        *info = -9;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -11;
    } else if (lwork < lwmin && !lquery) {
        *info = -13;
    }

    if (*info == 0) {
        work[0] = (double complex)lwmin;
    }

    if (*info != 0) {
        xerbla("ZGEMLQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmnk == 0) {
        return;
    }

    /* Choose between ZGEMLQT (standard) and ZLAMSWLQ (short-wide) */
    if ((left && m <= k) || (right && n <= k)
        || (nb <= k) || (nb >= (m > n ? (m > k ? m : k) : (n > k ? n : k)))) {
        /* Use standard blocked algorithm */
        zgemlqt(side, trans, m, n, k, mb,
                A, lda, &T[5], mb, C, ldc, work, info);
    } else {
        /* Use short-wide algorithm */
        zlamswlq(side, trans, m, n, k, mb, nb,
                 A, lda, &T[5], mb, C, ldc, work, lwork, info);
    }

    work[0] = (double complex)lwmin;
}
