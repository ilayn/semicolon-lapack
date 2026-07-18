/**
 * @file dggqrf.c
 * @brief DGGQRF computes a generalized QR factorization of a pair of matrices.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DGGQRF computes a generalized QR factorization of an `n`-by-`m` matrix `A`
 * and an `n`-by-`p` matrix `B`: `A = Q*R`, `B = Q*T*Z`, where Q is an
 * `n`-by-`n` orthogonal matrix, Z is a `p`-by-`p` orthogonal matrix, and R
 * and T assume one of the forms:
 * @rst
 * .. code-block:: text
 *
 *     if N >= M,  R = ( R11 ) M  ,   or if N < M,  R = ( R11  R12 ) N,
 *                     (  0  ) N-M                         N   M-N
 *                        M
 * @endrst
 * where R11 is upper triangular, and
 * @rst
 * .. code-block:: text
 *
 *     if N <= P,  T = ( 0  T12 ) N,   or if N > P,  T = ( T11 ) N-P,
 *                      P-N  N                           ( T21 ) P
 *                                                          P
 * @endrst
 * where T12 or T21 is upper triangular.
 *
 * In particular, if B is square and nonsingular, the GQR factorization of A
 * and B implicitly gives the QR factorization of `inv(B)*A`:
 * `inv(B)*A = Z**T*(inv(T)*R)` where `inv(B)` denotes the inverse of the
 * matrix B, and `Z**T` denotes the transpose of the matrix Z.
 *
 * @param[in]     n     The number of rows of the matrices `A` and `B`. `n>=0`.
 * @param[in]     m     The number of columns of the matrix `A`. `m>=0`.
 * @param[in]     p     The number of columns of the matrix `B`. `p>=0`.
 * @param[in,out] A     On entry, the `n`-by-`m` matrix `A`. On exit, the elements on
 *                      and above the diagonal of the array contain the `min(n,m)`-by-`m`
 *                      upper trapezoidal matrix R (R is upper triangular if `n>=m`);
 *                      the elements below the diagonal, with the array `taua`, represent
 *                      the orthogonal matrix Q as a product of `min(n,m)` elementary
 *                      reflectors (see Further Details).
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[out]    taua  Array of dimension (`min(n,m)`). The scalar factors of the
 *                      elementary reflectors which represent the orthogonal matrix Q
 *                      (see Further Details).
 * @param[in,out] B     On entry, the `n`-by-`p` matrix `B`. On exit, if `n<=p`, the
 *                      upper triangle of the subarray `B[0:n-1, p-n:p-1]` contains the
 *                      `n`-by-`n` upper triangular matrix T; if `n>p`, the elements on
 *                      and above the (n-p)-th subdiagonal contain the `n`-by-`p` upper
 *                      trapezoidal matrix T; the remaining elements, with the array
 *                      `taub`, represent the orthogonal matrix Z as a product of
 *                      elementary reflectors (see Further Details).
 * @param[in]     ldb   The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    taub  Array of dimension (`min(n,p)`). The scalar factors of the
 *                      elementary reflectors which represent the orthogonal matrix Z
 *                      (see Further Details).
 * @param[out]    work  Array of dimension (`max(1,lwork)`). On exit, if `info=0`,
 *                      `work[0]` returns the optimal `lwork`.
 * @param[in]     lwork The dimension of the array `work`. `lwork>=max(1,n,m,p)`.
 *                      For optimum performance `lwork>=max(n,m,p)*max(NB1,NB2,NB3)`,
 *                      where NB1 is the optimal blocksize for the QR factorization of an
 *                      `n`-by-`m` matrix, NB2 is the optimal blocksize for the RQ
 *                      factorization of an `n`-by-`p` matrix, and NB3 is the optimal
 *                      blocksize for a call of DORMQR.
 *                      If `lwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the `work` array, returns
 *                      this value as the first entry of the `work` array.
 * @param[out]    info  `info=0`: successful exit
 *                      `info<0`: if `info=-i`, the i-th argument had an illegal value.
 *
 * @par Further Details:
 * @rst
 * The matrix Q is represented as a product of elementary reflectors
 *
 * .. code-block:: text
 *
 *     Q = H(0) H(1) . . . H(k-1), where k = min(n,m).
 *
 *     Each H(i) has the form
 *
 *     H(i) = I - taua * v * v**T
 *
 *     where taua is a real scalar, and v is a real vector with
 *     v[0:i-1] = 0 and v[i] = 1; v[i+1:n-1] is stored on exit in
 *     A[i+1:n-1, i], and taua in taua[i].
 *
 * To form Q explicitly, use LAPACK subroutine DORGQR.
 * To use Q to update another matrix, use LAPACK subroutine DORMQR.
 *
 * The matrix Z is represented as a product of elementary reflectors
 *
 * .. code-block:: text
 *
 *     Z = H(0) H(1) . . . H(k-1), where k = min(n,p).
 *
 *     Each H(i) has the form
 *
 *     H(i) = I - taub * v * v**T
 *
 *     where taub is a real scalar, and v is a real vector with
 *     v[p-k+i+1:p-1] = 0 and v[p-k+i] = 1; v[0:p-k+i-1] is stored on exit in
 *     B[n-k+i, 0:p-k+i-1], and taub in taub[i].
 *
 * To form Z explicitly, use LAPACK subroutine DORGRQ.
 * To use Z to update another matrix, use LAPACK subroutine DORMRQ.
 * @endrst
 */
void dggqrf(const INT n, const INT m, const INT p,
            f64* restrict A, const INT lda, f64* restrict taua,
            f64* restrict B, const INT ldb, f64* restrict taub,
            f64* restrict work, const INT lwork, INT* info)
{
    INT lquery, nb, nb1, nb2, nb3, lwkopt, lopt;
    INT minval;

    *info = 0;
    nb1 = lapack_get_nb("GEQRF");
    nb2 = lapack_get_nb("GERQF");
    nb3 = lapack_get_nb("ORMQR");
    nb = nb1;
    if (nb2 > nb) nb = nb2;
    if (nb3 > nb) nb = nb3;

    minval = n;
    if (m > minval) minval = m;
    if (p > minval) minval = p;
    if (minval < 1) minval = 1;

    lwkopt = minval * nb;
    if (lwkopt < 1) lwkopt = 1;
    work[0] = (f64)lwkopt;

    lquery = (lwork == -1);

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (lda < 1 || lda < n) {
        *info = -5;
    } else if (ldb < 1 || ldb < n) {
        *info = -8;
    } else if (lwork < minval && !lquery) {
        *info = -11;
    }

    if (*info != 0) {
        xerbla("DGGQRF", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* QR factorization of N-by-M matrix A: A = Q*R */
    dgeqrf(n, m, A, lda, taua, work, lwork, info);
    lopt = (INT)work[0];

    /* Update B := Q**T * B */
    {
        INT minmn = (n < m) ? n : m;
        dormqr("L", "T", n, p, minmn, A, lda, taua, B, ldb, work, lwork, info);
    }
    if ((INT)work[0] > lopt) lopt = (INT)work[0];

    /* RQ factorization of N-by-P matrix B: B = T*Z */
    dgerqf(n, p, B, ldb, taub, work, lwork, info);
    if ((INT)work[0] > lopt) lopt = (INT)work[0];

    work[0] = (f64)lopt;
}
