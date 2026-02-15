/**
 * @file cggqrf.c
 * @brief CGGQRF computes a generalized QR factorization of a pair of matrices.
 */

#include <complex.h>
#include "semicolon_lapack_complex_single.h"
#include "../include/lapack_tuning.h"

void cggqrf(const int n, const int m, const int p,
            c64* restrict A, const int lda, c64* restrict taua,
            c64* restrict B, const int ldb, c64* restrict taub,
            c64* restrict work, const int lwork, int* info)
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
    work[0] = (c64)lwkopt;

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
        xerbla("CGGQRF", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* QR factorization of N-by-M matrix A: A = Q*R */
    cgeqrf(n, m, A, lda, taua, work, lwork, info);
    lopt = (int)crealf(work[0]);

    /* Update B := Q**H * B */
    {
        int minmn = (n < m) ? n : m;
        cunmqr("L", "C", n, p, minmn, A, lda, taua, B, ldb, work, lwork, info);
    }
    if ((int)crealf(work[0]) > lopt) lopt = (int)crealf(work[0]);

    /* RQ factorization of N-by-P matrix B: B = T*Z */
    cgerqf(n, p, B, ldb, taub, work, lwork, info);
    if ((int)crealf(work[0]) > lopt) lopt = (int)crealf(work[0]);

    work[0] = (c64)lopt;
}
