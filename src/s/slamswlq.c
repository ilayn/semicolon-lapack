/**
 * @file slamswlq.c
 * @brief SLAMSWLQ overwrites the general real M-by-N matrix C with
 *        Q*C, Q^T*C, C*Q, or C*Q^T using the Short-Wide LQ factorization.
 */

#include "semicolon_lapack_single.h"

/**
 * SLAMSWLQ overwrites the general real M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'T':   Q^T * C        C * Q^T
 *
 * where Q is a real orthogonal matrix defined as the product of blocked
 * elementary reflectors computed by short wide LQ factorization (SLASWLQ)
 *
 * @param[in]     side   'L': apply Q or Q^T from the Left;
 *                       'R': apply Q or Q^T from the Right.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'T': Transpose, apply Q^T.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q. m >= k >= 0.
 * @param[in]     mb     The row block size to be used in the blocked LQ.
 *                       m >= mb >= 1.
 * @param[in]     nb     The column block size to be used in the blocked LQ.
 *                       nb > m.
 * @param[in]     A      Double precision array, dimension (lda, m) if SIDE='L',
 *                       (lda, n) if SIDE='R'.
 *                       The i-th row must contain the vector which defines the
 *                       blocked elementary reflector H(i), as returned by SLASWLQ.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, k).
 * @param[in]     T      Double precision array, dimension
 *                       (m * Number of blocks(CEIL(N-K/NB-K))).
 *                       The blocked upper triangular block reflectors stored
 *                       in compact form.
 * @param[in]     ldt    The leading dimension of the array T. ldt >= max(1, mb).
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C, Q^T*C, C*Q^T, or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Double precision workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the minimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If min(m,n,k) = 0, lwork >= 1.
 *                       If SIDE = 'L', lwork >= max(1, nb*mb).
 *                       If SIDE = 'R', lwork >= max(1, m*mb).
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void slamswlq(const char* side, const char* trans,
              const int m, const int n, const int k,
              const int mb, const int nb,
              const float* const restrict A, const int lda,
              const float* const restrict T, const int ldt,
              float* const restrict C, const int ldc,
              float* const restrict work, const int lwork,
              int* info)
{
    int left, right, tran, notran, lquery;
    int i, ii, kk, ctr, lw, minmnk, lwmin;

    /* Decode arguments */
    lquery = (lwork == -1);
    notran = (trans[0] == 'N' || trans[0] == 'n');
    tran   = (trans[0] == 'T' || trans[0] == 't');
    left   = (side[0] == 'L' || side[0] == 'l');
    right  = (side[0] == 'R' || side[0] == 'r');

    if (left) {
        lw = n * mb;
    } else {
        lw = m * mb;
    }

    minmnk = m < n ? m : n;
    minmnk = minmnk < k ? minmnk : k;
    if (minmnk == 0) {
        lwmin = 1;
    } else {
        lwmin = lw > 1 ? lw : 1;
    }

    /* Parameter validation */
    *info = 0;
    if (!left && !right) {
        *info = -1;
    } else if (!tran && !notran) {
        *info = -2;
    } else if (k < 0) {
        *info = -5;
    } else if (m < k) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < mb || mb < 1) {
        *info = -6;
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -9;
    } else if (ldt < (mb > 1 ? mb : 1)) {
        *info = -11;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -13;
    } else if (lwork < lwmin && !lquery) {
        *info = -15;
    }

    if (*info == 0) {
        work[0] = (float)lwmin;
    }
    if (*info != 0) {
        xerbla("SLAMSWLQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmnk == 0) {
        return;
    }

    if ((nb <= k) || (nb >= (m > n ? (m > k ? m : k) : (n > k ? n : k)))) {
        sgemlqt(side, trans, m, n, k, mb, A, lda, T, ldt, C, ldc, work, info);
        return;
    }

    if (left && tran) {
        /* Case: Q^T * C -- backward loop from last block to first */

        kk = (m - k) % (nb - k);
        ctr = (m - k) / (nb - k);
        if (kk > 0) {
            ii = m - kk;
            stpmlqt("L", "T", kk, n, k, 0, mb,
                    &A[0 + ii * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[ii + 0 * ldc], ldc, work, info);
        } else {
            ii = m;
        }

        for (i = ii - (nb - k); i >= nb; i -= (nb - k)) {
            ctr = ctr - 1;
            stpmlqt("L", "T", nb - k, n, k, 0, mb,
                    &A[0 + i * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[i + 0 * ldc], ldc, work, info);
        }

        sgemlqt("L", "T", nb, n, k, mb, &A[0 + 0 * lda], lda, T, ldt,
                &C[0 + 0 * ldc], ldc, work, info);

    } else if (left && notran) {
        /* Case: Q * C -- forward loop from first block to last */

        kk = (m - k) % (nb - k);
        ii = m - kk;
        ctr = 1;

        sgemlqt("L", "N", nb, n, k, mb, &A[0 + 0 * lda], lda, T, ldt,
                &C[0 + 0 * ldc], ldc, work, info);

        for (i = nb; i <= ii - nb + k; i += (nb - k)) {
            stpmlqt("L", "N", nb - k, n, k, 0, mb,
                    &A[0 + i * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[i + 0 * ldc], ldc, work, info);
            ctr = ctr + 1;
        }

        if (ii < m) {
            stpmlqt("L", "N", kk, n, k, 0, mb,
                    &A[0 + ii * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[ii + 0 * ldc], ldc, work, info);
        }

    } else if (right && notran) {
        /* Case: C * Q -- backward loop from last block to first */

        kk = (n - k) % (nb - k);
        ctr = (n - k) / (nb - k);
        if (kk > 0) {
            ii = n - kk;
            stpmlqt("R", "N", m, kk, k, 0, mb,
                    &A[0 + ii * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[0 + ii * ldc], ldc, work, info);
        } else {
            ii = n;
        }

        for (i = ii - (nb - k); i >= nb; i -= (nb - k)) {
            ctr = ctr - 1;
            stpmlqt("R", "N", m, nb - k, k, 0, mb,
                    &A[0 + i * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[0 + i * ldc], ldc, work, info);
        }

        sgemlqt("R", "N", m, nb, k, mb, &A[0 + 0 * lda], lda, T, ldt,
                &C[0 + 0 * ldc], ldc, work, info);

    } else if (right && tran) {
        /* Case: C * Q^T -- forward loop from first block to last */

        kk = (n - k) % (nb - k);
        ctr = 1;
        ii = n - kk;

        sgemlqt("R", "T", m, nb, k, mb, &A[0 + 0 * lda], lda, T, ldt,
                &C[0 + 0 * ldc], ldc, work, info);

        for (i = nb; i <= ii - nb + k; i += (nb - k)) {
            stpmlqt("R", "T", m, nb - k, k, 0, mb,
                    &A[0 + i * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[0 + i * ldc], ldc, work, info);
            ctr = ctr + 1;
        }

        if (ii < n) {
            stpmlqt("R", "T", m, kk, k, 0, mb,
                    &A[0 + ii * lda], lda,
                    &T[0 + ctr * k * ldt], ldt,
                    &C[0 + 0 * ldc], ldc,
                    &C[0 + ii * ldc], ldc, work, info);
        }
    }

    work[0] = (float)lwmin;
}
