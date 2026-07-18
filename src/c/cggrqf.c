/**
 * @file cggrqf.c
 * @brief CGGRQF computes a generalized RQ factorization of a pair of matrices.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * CGGRQF computes a generalized RQ factorization of an `m`-by-`n` matrix `A`
 * and a `p`-by-`n` matrix `B`: `A = R*Q`, `B = Z*T*Q`, where Q is an
 * `n`-by-`n` unitary matrix, Z is a `p`-by-`p` unitary matrix, and R
 * and T assume one of the forms:
 * @rst
 * .. code-block:: text
 *
 *     if M <= N,  R = ( 0  R12 ) M,   or if M > N,  R = ( R11 ) M-N,
 *                      N-M  M                           ( R21 ) N
 *                                                          N
 * @endrst
 * where R12 or R21 is upper triangular, and
 * @rst
 * .. code-block:: text
 *
 *     if P >= N,  T = ( T11 ) N  ,   or if P < N,  T = ( T11  T12 ) P,
 *                     (  0  ) P-N                         P   N-P
 *                        N
 * @endrst
 * where T11 is upper triangular.
 *
 * In particular, if B is square and nonsingular, the GRQ factorization of A
 * and B implicitly gives the RQ factorization of `A*inv(B)`:
 * `A*inv(B) = (R*inv(T))*Z**H` where `inv(B)` denotes the inverse of the
 * matrix B, and `Z**H` denotes the conjugate transpose of the matrix Z.
 *
 * @param[in]     m     The number of rows of the matrix `A`. `m>=0`.
 * @param[in]     p     The number of rows of the matrix `B`. `p>=0`.
 * @param[in]     n     The number of columns of the matrices `A` and `B`. `n>=0`.
 * @param[in,out] A     On entry, the `m`-by-`n` matrix `A`. On exit, if `m<=n`, the
 *                      upper triangle of the subarray `A[0:m-1, n-m:n-1]` contains the
 *                      `m`-by-`m` upper triangular matrix R; if `m>n`, the elements on
 *                      and above the (m-n)-th subdiagonal contain the `m`-by-`n` upper
 *                      trapezoidal matrix R; the remaining elements, with the array
 *                      `taua`, represent the unitary matrix Q as a product of
 *                      elementary reflectors (see Further Details).
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,m)`.
 * @param[out]    taua  Array of dimension (`min(m,n)`). The scalar factors of the
 *                      elementary reflectors which represent the unitary matrix Q
 *                      (see Further Details).
 * @param[in,out] B     On entry, the `p`-by-`n` matrix `B`. On exit, the elements on
 *                      and above the diagonal of the array contain the `min(p,n)`-by-`n`
 *                      upper trapezoidal matrix T (T is upper triangular if `p>=n`);
 *                      the elements below the diagonal, with the array `taub`, represent
 *                      the unitary matrix Z as a product of elementary reflectors
 *                      (see Further Details).
 * @param[in]     ldb   The leading dimension of the array `B`. `ldb>=max(1,p)`.
 * @param[out]    taub  Array of dimension (`min(p,n)`). The scalar factors of the
 *                      elementary reflectors which represent the unitary matrix Z
 *                      (see Further Details).
 * @param[out]    work  Array of dimension (`max(1,lwork)`). On exit, if `info=0`,
 *                      `work[0]` returns the optimal `lwork`.
 * @param[in]     lwork The dimension of the array `work`. `lwork>=max(1,n,m,p)`.
 *                      For optimum performance `lwork>=max(n,m,p)*max(NB1,NB2,NB3)`,
 *                      where NB1 is the optimal blocksize for the RQ factorization of an
 *                      `m`-by-`n` matrix, NB2 is the optimal blocksize for the QR
 *                      factorization of a `p`-by-`n` matrix, and NB3 is the optimal
 *                      blocksize for a call of CUNMRQ.
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
 *     Q = H(0) H(1) . . . H(k-1), where k = min(m,n).
 *
 *     Each H(i) has the form
 *
 *     H(i) = I - taua * v * v**H
 *
 *     where taua is a complex scalar, and v is a complex vector with
 *     v[n-k+i+1:n-1] = 0 and v[n-k+i] = 1; v[0:n-k+i-1] is stored on exit in
 *     A[m-k+i, 0:n-k+i-1], and taua in taua[i].
 *
 * To form Q explicitly, use LAPACK subroutine CUNGRQ.
 * To use Q to update another matrix, use LAPACK subroutine CUNMRQ.
 *
 * The matrix Z is represented as a product of elementary reflectors
 *
 * .. code-block:: text
 *
 *     Z = H(0) H(1) . . . H(k-1), where k = min(p,n).
 *
 *     Each H(i) has the form
 *
 *     H(i) = I - taub * v * v**H
 *
 *     where taub is a complex scalar, and v is a complex vector with
 *     v[0:i-1] = 0 and v[i] = 1; v[i+1:p-1] is stored on exit in
 *     B[i+1:p-1, i], and taub in taub[i].
 *
 * To form Z explicitly, use LAPACK subroutine CUNGQR.
 * To use Z to update another matrix, use LAPACK subroutine CUNMQR.
 * @endrst
 */
void cggrqf(const INT m, const INT p, const INT n,
            c64* restrict A, const INT lda, c64* restrict taua,
            c64* restrict B, const INT ldb, c64* restrict taub,
            c64* restrict work, const INT lwork, INT* info)
{
    INT lquery, nb, nb1, nb2, nb3, lwkopt, lopt;
    INT minval, arow;

    *info = 0;
    nb1 = lapack_get_nb("GERQF");
    nb2 = lapack_get_nb("GEQRF");
    nb3 = lapack_get_nb("ORMRQ");
    nb = nb1;
    if (nb2 > nb) nb = nb2;
    if (nb3 > nb) nb = nb3;

    minval = n;
    if (m > minval) minval = m;
    if (p > minval) minval = p;
    if (minval < 1) minval = 1;

    lwkopt = minval * nb;
    if (lwkopt < 1) lwkopt = 1;
    work[0] = (c64)lwkopt;

    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (p < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < 1 || lda < m) {
        *info = -5;
    } else if (ldb < 1 || ldb < p) {
        *info = -8;
    } else if (lwork < minval && !lquery) {
        *info = -11;
    }

    if (*info != 0) {
        xerbla("CGGRQF", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* RQ factorization of M-by-N matrix A: A = R*Q */
    cgerqf(m, n, A, lda, taua, work, lwork, info);
    lopt = (INT)crealf(work[0]);

    /* Update B := B * Q**H */
    {
        INT minmn = (m < n) ? m : n;
        /* A(max(1, m-n+1), 1) in Fortran -> A[(m-n > 0 ? m-n : 0) * lda] in C */
        arow = (m - n > 0) ? (m - n) : 0;
        cunmrq("R", "C", p, n, minmn, &A[arow], lda, taua, B, ldb, work, lwork, info);
    }
    if ((INT)crealf(work[0]) > lopt) lopt = (INT)crealf(work[0]);

    /* QR factorization of P-by-N matrix B: B = Z*T */
    cgeqrf(p, n, B, ldb, taub, work, lwork, info);
    if ((INT)crealf(work[0]) > lopt) lopt = (INT)crealf(work[0]);

    work[0] = (c64)lopt;
}
