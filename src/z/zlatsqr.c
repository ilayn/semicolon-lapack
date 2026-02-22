/**
 * @file zlatsqr.c
 * @brief ZLATSQR computes a blocked Tall-Skinny QR factorization of
 *        a complex M-by-N matrix A for M >= N.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLATSQR computes a blocked Tall-Skinny QR factorization of
 * a complex M-by-N matrix A for M >= N:
 *
 *    A = Q * ( R ),
 *            ( 0 )
 *
 * where:
 *
 *    Q is a M-by-M orthogonal matrix, stored on exit in an implicit
 *    form in the elements below the diagonal of the array A and in
 *    the elements of the array T;
 *
 *    R is an upper-triangular N-by-N matrix, stored on exit in
 *    the elements on and above the diagonal of the array A.
 *
 *    0 is a (M-N)-by-N zero matrix, and is not stored.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. m >= n >= 0.
 * @param[in]     mb     The row block size to be used in the blocked QR. mb > 0.
 * @param[in]     nb     The column block size to be used in the blocked QR.
 *                       n >= nb >= 1.
 * @param[in,out] A      Double complex array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, the elements on and above the diagonal
 *                       of the array contain the N-by-N upper triangular matrix R;
 *                       the elements below the diagonal represent Q by the columns
 *                       of blocked V.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T      Double complex array,
 *                       dimension (ldt, n * Number_of_row_blocks)
 *                       where Number_of_row_blocks = CEIL((m-n)/(mb-n)).
 *                       The blocked upper triangular block reflectors stored in
 *                       compact form as a sequence of upper triangular blocks.
 * @param[in]     ldt    The leading dimension of the array T. ldt >= nb.
 * @param[out]    work   Workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the minimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       lwork >= 1, if min(m,n) = 0, and lwork >= nb*n, otherwise.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zlatsqr(const INT m, const INT n, const INT mb, const INT nb,
             c128* restrict A, const INT lda,
             c128* restrict T, const INT ldt,
             c128* restrict work, const INT lwork,
             INT* info)
{
    INT lquery;
    INT i, ii, kk, ctr, minmn, lwmin;

    *info = 0;

    lquery = (lwork == -1);

    minmn = m < n ? m : n;
    if (minmn == 0) {
        lwmin = 1;
    } else {
        lwmin = n * nb;
    }

    if (m < 0) {
        *info = -1;
    } else if (n < 0 || m < n) {
        *info = -2;
    } else if (mb < 1) {
        *info = -3;
    } else if (nb < 1 || (nb > n && n > 0)) {
        *info = -4;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -6;
    } else if (ldt < nb) {
        *info = -8;
    } else if (lwork < lwmin && !lquery) {
        *info = -10;
    }

    if (*info == 0) {
        work[0] = (c128)lwmin;
    }

    if (*info != 0) {
        xerbla("ZLATSQR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (minmn == 0) {
        return;
    }

    /* The QR Decomposition */
    if ((mb <= n) || (mb >= m)) {
        zgeqrt(m, n, nb, A, lda, T, ldt, work, info);
        return;
    }

    kk = (m - n) % (mb - n);
    ii = m - kk;  /* 0-based: Fortran II = M-KK+1 becomes C ii = m - kk */

    /* Compute the QR factorization of the first block A(0:mb-1, 0:n-1) */
    zgeqrt(mb, n, nb, &A[0], lda, T, ldt, work, info);

    ctr = 1;
    for (i = mb; i <= ii - mb + n; i += (mb - n)) {
        /* Compute the QR factorization of the current block A(i:i+mb-n-1, 0:n-1) */
        ztpqrt(mb - n, n, 0, nb, &A[0], lda, &A[i], lda,
               &T[0 + ctr * n * ldt], ldt, work, info);
        ctr = ctr + 1;
    }

    /* Compute the QR factorization of the last block A(ii:m-1, 0:n-1) */
    if (ii < m) {
        ztpqrt(kk, n, 0, nb, &A[0], lda, &A[ii], lda,
               &T[0 + ctr * n * ldt], ldt, work, info);
    }

    work[0] = (c128)lwmin;
}
