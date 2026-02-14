/**
 * @file dgelq.c
 * @brief DGELQ computes an LQ factorization of a real M-by-N matrix A.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DGELQ computes an LQ factorization of a real M-by-N matrix A:
 *
 *    A = ( L 0 ) *  Q
 *
 * where:
 *
 *    Q is a N-by-N orthogonal matrix;
 *    L is a lower-triangular M-by-M matrix;
 *    0 is a M-by-(N-M) zero matrix, if M < N.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, the elements on and below the diagonal
 *                       contain the M-by-min(M,N) lower trapezoidal matrix L;
 *                       the elements above the diagonal are used to store
 *                       part of the data structure to represent Q.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T      Double precision array, dimension (max(5, tsize)).
 *                       On exit, if info = 0, T[0] returns optimal (or minimal)
 *                       tsize. Remaining T contains part of the data structure
 *                       used to represent Q.
 * @param[in]     tsize  If tsize >= 5, the dimension of the array T.
 *                       If tsize = -1 or -2, then a workspace query is assumed.
 *                       If tsize = -1, calculates optimal size for T.
 *                       If tsize = -2, calculates minimal size for T.
 * @param[out]    work   Double precision workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] contains optimal (or minimal)
 *                       lwork.
 * @param[in]     lwork  The dimension of the array work. lwork >= 1.
 *                       If lwork = -1 or -2, then a workspace query is assumed.
 *                       If lwork = -1, calculates optimal size for work.
 *                       If lwork = -2, calculates minimal size for work.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgelq(const int m, const int n,
           f64* restrict A, const int lda,
           f64* restrict T, const int tsize,
           f64* restrict work, const int lwork,
           int* info)
{
    int lquery, lminws, mint, minw;
    int mb, nb, mintsz, nblcks, lwmin, lwopt, lwreq;
    int minmn;

    *info = 0;

    lquery = (tsize == -1 || tsize == -2 || lwork == -1 || lwork == -2);

    mint = 0;
    minw = 0;
    if (tsize == -2 || lwork == -2) {
        if (tsize != -1) mint = 1;
        if (lwork != -1) minw = 1;
    }

    /* Determine the block size */
    minmn = m < n ? m : n;
    if (minmn > 0) {
        mb = lapack_get_gelq_mb(m, n);
        nb = lapack_get_gelq_nb(m, n);
    } else {
        mb = 1;
        nb = n;
    }
    if (mb > minmn || mb < 1) mb = 1;
    if (nb > n || nb <= m) nb = n;
    mintsz = m + 5;

    if (nb > m && n > m) {
        if ((n - m) % (nb - m) == 0) {
            nblcks = (n - m) / (nb - m);
        } else {
            nblcks = (n - m) / (nb - m) + 1;
        }
    } else {
        nblcks = 1;
    }

    /* Determine if the workspace size satisfies minimal size */
    if ((n <= m) || (nb <= m) || (nb >= n)) {
        lwmin = n > 1 ? n : 1;
        lwopt = mb * n > 1 ? mb * n : 1;
    } else {
        lwmin = m > 1 ? m : 1;
        lwopt = mb * m > 1 ? mb * m : 1;
    }

    lminws = 0;
    if ((tsize < (mb * m * nblcks + 5 > 1 ? mb * m * nblcks + 5 : 1) || lwork < lwopt)
        && (lwork >= lwmin) && (tsize >= mintsz)
        && (!lquery)) {
        if (tsize < (mb * m * nblcks + 5 > 1 ? mb * m * nblcks + 5 : 1)) {
            lminws = 1;
            mb = 1;
            nb = n;
        }
        if (lwork < lwopt) {
            lminws = 1;
            mb = 1;
        }
    }

    if ((n <= m) || (nb <= m) || (nb >= n)) {
        lwreq = mb * n > 1 ? mb * n : 1;
    } else {
        lwreq = mb * m > 1 ? mb * m : 1;
    }

    /* Parameter validation */
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    } else if (tsize < (mb * m * nblcks + 5 > 1 ? mb * m * nblcks + 5 : 1)
               && !lquery && !lminws) {
        *info = -6;
    } else if (lwork < lwreq && !lquery && !lminws) {
        *info = -8;
    }

    if (*info == 0) {
        if (mint) {
            T[0] = (f64)mintsz;
        } else {
            T[0] = (f64)(mb * m * nblcks + 5);
        }
        T[1] = (f64)mb;
        T[2] = (f64)nb;
        if (minw) {
            work[0] = (f64)lwmin;
        } else {
            work[0] = (f64)lwreq;
        }
    }
    if (*info != 0) {
        xerbla("DGELQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmn == 0) {
        return;
    }

    /* The LQ Decomposition */
    if ((n <= m) || (nb <= m) || (nb >= n)) {
        dgelqt(m, n, mb, A, lda, &T[5], mb, work, info);
    } else {
        dlaswlq(m, n, mb, nb, A, lda, &T[5], mb, work, lwork, info);
    }

    work[0] = (f64)lwreq;
}
