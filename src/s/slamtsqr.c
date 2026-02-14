/**
 * @file slamtsqr.c
 * @brief SLAMTSQR overwrites the general real M-by-N matrix C with
 *        Q*C, Q^T*C, C*Q, or C*Q^T using the blocked reflectors from SLATSQR.
 */

#include "semicolon_lapack_single.h"

/**
 * SLAMTSQR overwrites the general real M-by-N matrix C with
 *
 *              SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':   Q * C          C * Q
 * TRANS = 'T':   Q^T * C        C * Q^T
 *
 * where Q is a real orthogonal matrix defined as the product
 * of blocked elementary reflectors computed by tall skinny
 * QR factorization (SLATSQR)
 *
 * @param[in]     side   'L': apply Q or Q^T from the Left;
 *                       'R': apply Q or Q^T from the Right.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'T': Transpose, apply Q^T.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors. m >= k >= 0.
 * @param[in]     mb     The row block size used in the blocked QR. mb > n.
 * @param[in]     nb     The column block size. n >= nb >= 1.
 * @param[in]     A      Double precision array, dimension (lda, k).
 *                       The blocked elementary reflectors as returned by SLATSQR.
 * @param[in]     lda    The leading dimension of A.
 *                       If SIDE = 'L', lda >= max(1, m);
 *                       if SIDE = 'R', lda >= max(1, n).
 * @param[in]     T      Double precision array containing the block reflectors.
 * @param[in]     ldt    The leading dimension of T. ldt >= nb.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C, Q^T*C, C*Q^T, or C*Q.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace array, dimension (max(1, lwork)).
 * @param[in]     lwork  The dimension of work.
 *                       If SIDE = 'L', lwork >= max(1, n*nb).
 *                       If SIDE = 'R', lwork >= max(1, mb*nb).
 *                       If lwork = -1, workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void slamtsqr(const char* side, const char* trans,
              const int m, const int n, const int k, const int mb, const int nb,
              const f32* restrict A, const int lda,
              const f32* restrict T, const int ldt,
              f32* restrict C, const int ldc,
              f32* restrict work, const int lwork,
              int* info)
{
    int left, right, tran, notran, lquery;
    int i, ii, kk, lw, ctr, q, minmnk, lwmin;

    *info = 0;
    lquery = (lwork == -1);
    notran = (trans[0] == 'N' || trans[0] == 'n');
    tran   = (trans[0] == 'T' || trans[0] == 't');
    left   = (side[0] == 'L' || side[0] == 'l');
    right  = (side[0] == 'R' || side[0] == 'r');

    if (left) {
        lw = n * nb;
        q = m;
    } else {
        lw = mb * nb;
        q = n;
    }

    minmnk = m < n ? m : n;
    minmnk = minmnk < k ? minmnk : k;

    if (minmnk == 0) {
        lwmin = 1;
    } else {
        lwmin = lw > 1 ? lw : 1;
    }

    if (!left && !right) {
        *info = -1;
    } else if (!tran && !notran) {
        *info = -2;
    } else if (m < k) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0) {
        *info = -5;
    } else if (k < nb || nb < 1) {
        *info = -7;
    } else if (lda < (q > 1 ? q : 1)) {
        *info = -9;
    } else if (ldt < (nb > 1 ? nb : 1)) {
        *info = -11;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -13;
    } else if (lwork < lwmin && !lquery) {
        *info = -15;
    }

    if (*info == 0) {
        work[0] = (f32)lwmin;
    }

    if (*info != 0) {
        xerbla("SLAMTSQR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmnk == 0) {
        return;
    }

    /* Determine if it is tall skinny or short and wide */
    {
        int maxmnk = m > n ? m : n;
        maxmnk = maxmnk > k ? maxmnk : k;
        if ((mb <= k) || (mb >= maxmnk)) {
            sgemqrt(side, trans, m, n, k, nb, A, lda, T, ldt, C, ldc, work, info);
            return;
        }
    }

    if (left && notran) {
        /*
         * Multiply Q to the last block of C
         */
        kk = (m - k) % (mb - k);
        ctr = (m - k) / (mb - k);
        if (kk > 0) {
            ii = m - kk;  /* 0-based: Fortran II=M-KK+1 */
            stpmqrt("L", "N", kk, n, k, 0, nb, &A[ii], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[ii], ldc, work, info);
        } else {
            ii = m;  /* 0-based: Fortran II=M+1 means ii=m (past end) */
        }

        /* Fortran: DO I=II-(MB-K),MB+1,-(MB-K) */
        /* C 0-based: i starts at ii-(mb-k), goes down to mb (0-based row mb), step -(mb-k) */
        for (i = ii - (mb - k); i >= mb; i -= (mb - k)) {
            /*
             * Multiply Q to the current block of C
             */
            ctr = ctr - 1;
            stpmqrt("L", "N", mb - k, n, k, 0, nb, &A[i], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[i], ldc, work, info);
        }

        /*
         * Multiply Q to the first block of C
         */
        sgemqrt("L", "N", mb, n, k, nb, &A[0], lda, T, ldt, &C[0], ldc, work, info);

    } else if (left && tran) {
        /*
         * Multiply Q^T to the first block of C
         */
        kk = (m - k) % (mb - k);
        ii = m - kk;  /* 0-based */
        ctr = 1;
        sgemqrt("L", "T", mb, n, k, nb, &A[0], lda, T, ldt, &C[0], ldc, work, info);

        /* Fortran: DO I=MB+1,II-MB+K,(MB-K) */
        /* C 0-based: i starts at mb, goes up to ii-mb+k-1, step (mb-k) */
        for (i = mb; i <= ii - mb + k; i += (mb - k)) {
            /*
             * Multiply Q^T to the current block of C
             */
            stpmqrt("L", "T", mb - k, n, k, 0, nb, &A[i], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[i], ldc, work, info);
            ctr = ctr + 1;
        }

        if (ii < m) {
            /*
             * Multiply Q^T to the last block of C
             */
            stpmqrt("L", "T", kk, n, k, 0, nb, &A[ii], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[ii], ldc, work, info);
        }

    } else if (right && tran) {
        /*
         * Multiply Q^T from the right to the last block of C
         */
        kk = (n - k) % (mb - k);
        ctr = (n - k) / (mb - k);
        if (kk > 0) {
            ii = n - kk;  /* 0-based */
            stpmqrt("R", "T", m, kk, k, 0, nb, &A[ii], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[0 + ii * ldc], ldc, work, info);
        } else {
            ii = n;  /* 0-based: past end */
        }

        /* Fortran: DO I=II-(MB-K),MB+1,-(MB-K) */
        for (i = ii - (mb - k); i >= mb; i -= (mb - k)) {
            /*
             * Multiply Q^T to the current block of C
             */
            ctr = ctr - 1;
            stpmqrt("R", "T", m, mb - k, k, 0, nb, &A[i], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[0 + i * ldc], ldc, work, info);
        }

        /*
         * Multiply Q^T to the first block of C
         */
        sgemqrt("R", "T", m, mb, k, nb, &A[0], lda, T, ldt, &C[0], ldc, work, info);

    } else if (right && notran) {
        /*
         * Multiply Q from the right to the first block of C
         */
        kk = (n - k) % (mb - k);
        ii = n - kk;  /* 0-based */
        ctr = 1;
        sgemqrt("R", "N", m, mb, k, nb, &A[0], lda, T, ldt, &C[0], ldc, work, info);

        /* Fortran: DO I=MB+1,II-MB+K,(MB-K) */
        for (i = mb; i <= ii - mb + k; i += (mb - k)) {
            /*
             * Multiply Q to the current block of C
             */
            stpmqrt("R", "N", m, mb - k, k, 0, nb, &A[i], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[0 + i * ldc], ldc, work, info);
            ctr = ctr + 1;
        }

        if (ii < n) {
            /*
             * Multiply Q to the last block of C
             */
            stpmqrt("R", "N", m, kk, k, 0, nb, &A[ii], lda,
                    &T[0 + ctr * k * ldt], ldt, &C[0], ldc,
                    &C[0 + ii * ldc], ldc, work, info);
        }
    }

    work[0] = (f32)lwmin;
}
