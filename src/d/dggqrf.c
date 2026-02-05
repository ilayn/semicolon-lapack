/**
 * @file dggqrf.c
 * @brief DGGQRF computes a generalized QR factorization of a pair of matrices.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

void dggqrf(const int n, const int m, const int p,
            double* const restrict A, const int lda, double* const restrict taua,
            double* const restrict B, const int ldb, double* const restrict taub,
            double* const restrict work, const int lwork, int* info)
{
    int lquery, nb, nb1, nb2, nb3, lwkopt, lopt;
    int minval;

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
    work[0] = (double)lwkopt;

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
    lopt = (int)work[0];

    /* Update B := Q**T * B */
    {
        int minmn = (n < m) ? n : m;
        dormqr("L", "T", n, p, minmn, A, lda, taua, B, ldb, work, lwork, info);
    }
    if ((int)work[0] > lopt) lopt = (int)work[0];

    /* RQ factorization of N-by-P matrix B: B = T*Z */
    dgerqf(n, p, B, ldb, taub, work, lwork, info);
    if ((int)work[0] > lopt) lopt = (int)work[0];

    work[0] = (double)lopt;
}
