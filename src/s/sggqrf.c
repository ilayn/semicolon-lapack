/**
 * @file sggqrf.c
 * @brief SGGQRF computes a generalized QR factorization of a pair of matrices.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

void sggqrf(const INT n, const INT m, const INT p,
            f32* restrict A, const INT lda, f32* restrict taua,
            f32* restrict B, const INT ldb, f32* restrict taub,
            f32* restrict work, const INT lwork, INT* info)
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
    work[0] = (f32)lwkopt;

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
        xerbla("SGGQRF", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* QR factorization of N-by-M matrix A: A = Q*R */
    sgeqrf(n, m, A, lda, taua, work, lwork, info);
    lopt = (INT)work[0];

    /* Update B := Q**T * B */
    {
        INT minmn = (n < m) ? n : m;
        sormqr("L", "T", n, p, minmn, A, lda, taua, B, ldb, work, lwork, info);
    }
    if ((INT)work[0] > lopt) lopt = (INT)work[0];

    /* RQ factorization of N-by-P matrix B: B = T*Z */
    sgerqf(n, p, B, ldb, taub, work, lwork, info);
    if ((INT)work[0] > lopt) lopt = (INT)work[0];

    work[0] = (f32)lopt;
}
