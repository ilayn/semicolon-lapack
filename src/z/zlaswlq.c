/**
 * @file zlaswlq.c
 * @brief ZLASWLQ computes a blocked Short-Wide LQ factorization of a
 *        complex M-by-N matrix A for M <= N.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLASWLQ computes a blocked Short-Wide LQ factorization of
 * a complex M-by-N matrix A for M <= N:
 *
 *    A = ( L 0 ) *  Q,
 *
 * where:
 *
 *    Q is a n-by-N orthogonal matrix, stored on exit in an implicit
 *    form in the elements above the diagonal of the array A and in
 *    the elements of the array T;
 *    L is a lower-triangular M-by-M matrix stored on exit in
 *    the elements on and below the diagonal of the array A.
 *    0 is a M-by-(N-M) zero matrix, if M < N, and is not stored.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= m >= 0.
 * @param[in]     mb     The row block size to be used in the blocked LQ.
 *                       m >= mb >= 1.
 * @param[in]     nb     The column block size to be used in the blocked LQ.
 *                       nb > 0.
 * @param[in,out] A      Double complex array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, the elements on and below the diagonal
 *                       contain the M-by-M lower triangular matrix L;
 *                       the elements above the diagonal represent Q by the rows
 *                       of blocked V.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T      Double complex array, dimension
 *                       (ldt, n * Number_of_row_blocks)
 *                       where Number_of_row_blocks = CEIL((N-M)/(NB-M)).
 *                       The blocked upper triangular block reflectors stored
 *                       in compact form as a sequence of upper triangular blocks.
 * @param[in]     ldt    The leading dimension of the array T. ldt >= mb.
 * @param[out]    work   Double complex workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the minimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       lwork >= 1, if min(m, n) = 0, and lwork >= mb*m, otherwise.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zlaswlq(const int m, const int n, const int mb, const int nb,
             double complex* const restrict A, const int lda,
             double complex* const restrict T, const int ldt,
             double complex* const restrict work, const int lwork,
             int* info)
{
    int lquery;
    int i, ii, kk, ctr, minmn, lwmin;

    *info = 0;

    lquery = (lwork == -1);

    minmn = m < n ? m : n;
    if (minmn == 0) {
        lwmin = 1;
    } else {
        lwmin = m * mb;
    }

    /* Parameter validation */
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n < m) {
        *info = -2;
    } else if (mb < 1 || (mb > m && m > 0)) {
        *info = -3;
    } else if (nb < 0) {
        *info = -4;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -6;
    } else if (ldt < mb) {
        *info = -8;
    } else if (lwork < lwmin && !lquery) {
        *info = -10;
    }

    if (*info == 0) {
        work[0] = (double complex)lwmin;
    }

    if (*info != 0) {
        xerbla("ZLASWLQ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmn == 0) {
        return;
    }

    /* The LQ Decomposition */
    if ((m >= n) || (nb <= m) || (nb >= n)) {
        zgelqt(m, n, mb, A, lda, T, ldt, work, info);
        return;
    }

    kk = (n - m) % (nb - m);
    ii = n - kk;

    /* Compute the LQ factorization of the first block A(0:m-1, 0:nb-1) */
    zgelqt(m, nb, mb, &A[0 + 0 * lda], lda, T, ldt, work, info);
    ctr = 1;

    for (i = nb; i <= ii - nb + m; i += (nb - m)) {
        /* Compute the LQ factorization of the current block A(0:m-1, i:i+nb-m-1) */
        ztplqt(m, nb - m, 0, mb, &A[0 + 0 * lda], lda, &A[0 + i * lda],
               lda, &T[0 + ctr * m * ldt], ldt, work, info);
        ctr = ctr + 1;
    }

    /* Compute the LQ factorization of the last block A(0:m-1, ii:n-1) */
    if (ii < n) {
        ztplqt(m, kk, 0, mb, &A[0 + 0 * lda], lda, &A[0 + ii * lda],
               lda, &T[0 + ctr * m * ldt], ldt, work, info);
    }

    work[0] = (double complex)lwmin;
}
