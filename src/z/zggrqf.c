/**
 * @file zggrqf.c
 * @brief ZGGRQF computes a generalized RQ factorization of a pair of matrices.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>

void zggrqf(const int m, const int p, const int n,
            c128* restrict A, const int lda, c128* restrict taua,
            c128* restrict B, const int ldb, c128* restrict taub,
            c128* restrict work, const int lwork, int* info)
{
    int lquery, nb, nb1, nb2, nb3, lwkopt, lopt;
    int minval, arow;

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
    work[0] = (c128)lwkopt;

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
        xerbla("ZGGRQF", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* RQ factorization of M-by-N matrix A: A = R*Q */
    zgerqf(m, n, A, lda, taua, work, lwork, info);
    lopt = (int)creal(work[0]);

    /* Update B := B * Q**H */
    {
        int minmn = (m < n) ? m : n;
        /* A(max(1, m-n+1), 1) in Fortran -> A[(m-n > 0 ? m-n : 0) * lda] in C */
        arow = (m - n > 0) ? (m - n) : 0;
        zunmrq("R", "C", p, n, minmn, A + arow, lda, taua, B, ldb, work, lwork, info);
    }
    if ((int)creal(work[0]) > lopt) lopt = (int)creal(work[0]);

    /* QR factorization of P-by-N matrix B: B = Z*T */
    zgeqrf(p, n, B, ldb, taub, work, lwork, info);
    if ((int)creal(work[0]) > lopt) lopt = (int)creal(work[0]);

    work[0] = (c128)lopt;
}
