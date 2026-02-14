/**
 * @file dgeqr.c
 * @brief DGEQR computes a QR factorization of a real M-by-N matrix A.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DGEQR computes a QR factorization of a real M-by-N matrix A:
 *
 *    A = Q * ( R ),
 *            ( 0 )
 *
 * where:
 *
 *    Q is a M-by-M orthogonal matrix;
 *    R is an upper-triangular N-by-N matrix;
 *    0 is a (M-N)-by-N zero matrix, if M > N.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, the elements on and above the diagonal contain
 *                       the min(m,n)-by-n upper trapezoidal matrix R; the elements
 *                       below the diagonal are used to store part of the data
 *                       structure to represent Q.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T      Double precision array, dimension (max(5, tsize)).
 *                       On exit, if info = 0, T[0] returns optimal (or minimal) tsize.
 *                       T[1] = MB, T[2] = NB (block sizes).
 *                       Remaining T contains part of the data structure for Q.
 * @param[in]     tsize  If tsize >= 5, the dimension of array T.
 *                       If tsize = -1 or -2, workspace query is assumed.
 * @param[out]    work   Workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] contains optimal (or minimal) lwork.
 * @param[in]     lwork  The dimension of array work. lwork >= 1.
 *                       If lwork = -1 or -2, workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dgeqr(const int m, const int n,
           f64* restrict A, const int lda,
           f64* restrict T, const int tsize,
           f64* restrict work, const int lwork,
           int* info)
{
    int lquery, lminws, mint, minw;
    int mb, nb, mintsz, nblcks, lwmin, lwreq;
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
        mb = lapack_get_geqr_mb(m, n);
        nb = lapack_get_geqr_nb(m, n);
    } else {
        mb = m;
        nb = 1;
    }
    if (mb > m || mb <= n) mb = m;
    if (nb > minmn || nb < 1) nb = 1;
    mintsz = n + 5;

    if (mb > n && m > n) {
        if ((m - n) % (mb - n) == 0) {
            nblcks = (m - n) / (mb - n);
        } else {
            nblcks = (m - n) / (mb - n) + 1;
        }
    } else {
        nblcks = 1;
    }

    /* Determine if the workspace size satisfies minimal size */
    lwmin = n > 1 ? n : 1;
    lwreq = n * nb > 1 ? n * nb : 1;
    lminws = 0;

    {
        int tsize_req = nb * n * nblcks + 5;
        tsize_req = tsize_req > 1 ? tsize_req : 1;
        if ((tsize < tsize_req || lwork < lwreq)
            && (lwork >= n) && (tsize >= mintsz)
            && (!lquery)) {
            if (tsize < tsize_req) {
                lminws = 1;
                nb = 1;
                mb = m;
            }
            if (lwork < lwreq) {
                lminws = 1;
                nb = 1;
            }
        }
    }

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    } else if (tsize < (nb * n * nblcks + 5 > 1 ? nb * n * nblcks + 5 : 1)
               && (!lquery) && (!lminws)) {
        *info = -6;
    } else if ((lwork < lwreq) && (!lquery) && (!lminws)) {
        *info = -8;
    }

    if (*info == 0) {
        if (mint) {
            T[0] = (f64)mintsz;
        } else {
            T[0] = (f64)(nb * n * nblcks + 5);
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
        xerbla("DGEQR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmn == 0) {
        return;
    }

    /* The QR Decomposition */
    if ((m <= n) || (mb <= n) || (mb >= m)) {
        dgeqrt(m, n, nb, A, lda, &T[5], nb, work, info);
    } else {
        dlatsqr(m, n, mb, nb, A, lda, &T[5], nb, work, lwork, info);
    }

    work[0] = (f64)lwreq;
}
