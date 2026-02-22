/**
 * @file sgelsy.c
 * @brief SGELSY computes the minimum-norm solution to a real linear least
 *        squares problem using complete orthogonal factorization.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * SGELSY computes the minimum-norm solution to a real linear least
 * squares problem:
 *     minimize || A * X - B ||
 * using a complete orthogonal factorization of A. A is an M-by-N
 * matrix which may be rank-deficient.
 *
 * The routine first computes a QR factorization with column pivoting:
 *     A * P = Q * [ R11 R12 ]
 *                 [  0  R22 ]
 * with R11 defined as the largest leading submatrix whose estimated
 * condition number is less than 1/RCOND. The order of R11, RANK,
 * is the effective rank of A.
 *
 * Then, R22 is considered to be negligible, and R12 is annihilated
 * by orthogonal transformations from the right, arriving at the
 * complete orthogonal factorization:
 *    A * P = Q * [ T11 0 ] * Z
 *                [  0  0 ]
 * The minimum-norm solution is then
 *    X = P * Z^T * [ inv(T11)*Q1^T*B ]
 *                  [        0         ]
 * where Q1 consists of the first RANK columns of Q.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the m-by-n matrix A.
 *                       On exit, overwritten by details of the complete
 *                       orthogonal factorization.
 * @param[in]     lda    Leading dimension of A. lda >= max(1, m).
 * @param[in,out] B      Double precision array, dimension (ldb, nrhs).
 *                       On entry, the m-by-nrhs right hand side matrix.
 *                       On exit, the n-by-nrhs solution matrix X.
 * @param[in]     ldb    Leading dimension of B. ldb >= max(1, m, n).
 * @param[in,out] jpvt   Integer array, dimension (n).
 *                       On entry, if jpvt[i] != 0, the i-th column of A is
 *                       permuted to the front of AP; otherwise column i is free.
 *                       On exit, if jpvt[i] = k, then the i-th column of A*P
 *                       was the k-th column of A (0-based).
 * @param[in]     rcond  Used to determine effective rank of A. Columns
 *                       whose condition number exceeds 1/rcond are treated
 *                       as zero.
 * @param[out]    rank   The effective rank of A, i.e., the order of R11.
 * @param[out]    work   Double precision array, dimension (max(1, lwork)).
 *                       On exit, work[0] returns the optimal lwork.
 * @param[in]     lwork  Dimension of work.
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgelsy(const INT m, const INT n, const INT nrhs,
            f32* restrict A, const INT lda,
            f32* restrict B, const INT ldb,
            INT* restrict jpvt, const f32 rcond,
            INT* rank,
            f32* restrict work, const INT lwork,
            INT* info)
{
    /* Constants from Fortran source: IMAX=1, IMIN=2 */
    const INT IMAX = 1;
    const INT IMIN = 2;

    INT lquery;
    INT iascl, ibscl, ismin, ismax, mn, nb;
    INT lwkmin, lwkopt;
    INT iinfo;
    f32 anrm, bignum, bnrm, smlnum;
    f32 c1, c2, s1, s2, smax, smaxpr, smin, sminpr;

    /* Initialization */
    mn = m < n ? m : n;
    ismin = mn;       /* 0-based offset: Fortran ISMIN = MN+1 */
    ismax = 2 * mn;   /* 0-based offset: Fortran ISMAX = 2*MN+1 */

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldb < (m > n ? (m > 1 ? m : 1) : (n > 1 ? n : 1))) {
        *info = -7;
    }

    /* Figure out optimal block size */
    if (*info == 0) {
        if (mn == 0 || nrhs == 0) {
            lwkmin = 1;
            lwkopt = 1;
        } else {
            INT nb1 = lapack_get_nb("GEQRF");
            INT nb2 = lapack_get_nb("GERQF");
            INT nb3 = lapack_get_nb("ORMQR");
            INT nb4 = lapack_get_nb("ORMRQ");
            nb = nb1;
            if (nb2 > nb) nb = nb2;
            if (nb3 > nb) nb = nb3;
            if (nb4 > nb) nb = nb4;

            /* LWKMIN = MN + MAX(2*MN, N+1, MN+NRHS) */
            INT t1 = 2 * mn;
            INT t2 = n + 1;
            INT t3 = mn + nrhs;
            INT tmax = t1;
            if (t2 > tmax) tmax = t2;
            if (t3 > tmax) tmax = t3;
            lwkmin = mn + tmax;

            /* LWKOPT = MAX(LWKMIN, MN+2*N+NB*(N+1), 2*MN+NB*NRHS) */
            INT w1 = mn + 2 * n + nb * (n + 1);
            INT w2 = 2 * mn + nb * nrhs;
            lwkopt = lwkmin;
            if (w1 > lwkopt) lwkopt = w1;
            if (w2 > lwkopt) lwkopt = w2;
        }
        work[0] = (f32)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("SGELSY", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (mn == 0 || nrhs == 0) {
        *rank = 0;
        return;
    }

    /* Get machine parameters */
    smlnum = slamch("S") / slamch("P");
    bignum = 1.0f / smlnum;

    /* Scale A, B if max entries outside range [SMLNUM, BIGNUM] */
    anrm = slange("M", m, n, A, lda, NULL);
    iascl = 0;
    if (anrm > 0.0f && anrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        slascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &iinfo);
        iascl = 1;
    } else if (anrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        slascl("G", 0, 0, anrm, bignum, m, n, A, lda, &iinfo);
        iascl = 2;
    } else if (anrm == 0.0f) {
        /* Matrix all zero. Return zero solution. */
        INT maxmn = m > n ? m : n;
        slaset("F", maxmn, nrhs, 0.0f, 0.0f, B, ldb);
        *rank = 0;
        work[0] = (f32)lwkopt;
        return;
    }

    bnrm = slange("M", m, nrhs, B, ldb, NULL);
    ibscl = 0;
    if (bnrm > 0.0f && bnrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        slascl("G", 0, 0, bnrm, smlnum, m, nrhs, B, ldb, &iinfo);
        ibscl = 1;
    } else if (bnrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        slascl("G", 0, 0, bnrm, bignum, m, nrhs, B, ldb, &iinfo);
        ibscl = 2;
    }

    /* Compute QR factorization with column pivoting of A:
     *   A * P = Q * R
     * tau stored in work[0..mn-1], sub-workspace in work[mn..] */
    sgeqp3(m, n, A, lda, jpvt, work, &work[mn], lwork - mn, &iinfo);

    /* Determine RANK using incremental condition estimation */
    work[ismin] = 1.0f;
    work[ismax] = 1.0f;
    smax = fabsf(A[0]);  /* |A(0,0)| = |R(1,1)| */
    smin = smax;

    if (fabsf(A[0]) == 0.0f) {
        *rank = 0;
        INT maxmn = m > n ? m : n;
        slaset("F", maxmn, nrhs, 0.0f, 0.0f, B, ldb);
        work[0] = (f32)lwkopt;
        return;
    } else {
        *rank = 1;
    }

    /* Rank determination loop (Fortran GO TO 10 pattern → while loop) */
    while (*rank < mn) {
        INT i = *rank;  /* 0-based column index: Fortran I = RANK+1 */
        slaic1(IMIN, *rank, &work[ismin], smin,
               &A[0 + i * lda], A[i + i * lda],
               &sminpr, &s1, &c1);
        slaic1(IMAX, *rank, &work[ismax], smax,
               &A[0 + i * lda], A[i + i * lda],
               &smaxpr, &s2, &c2);

        if (smaxpr * rcond > sminpr) {
            break;
        }

        for (INT ii = 0; ii < *rank; ii++) {
            work[ismin + ii] = s1 * work[ismin + ii];
            work[ismax + ii] = s2 * work[ismax + ii];
        }
        work[ismin + *rank] = c1;
        work[ismax + *rank] = c2;
        smin = sminpr;
        smax = smaxpr;
        (*rank)++;
    }

    /* workspace: 3*MN.
     *
     * Logically partition R = [ R11 R12 ]
     *                          [  0  R22 ]
     * where R11 = R(0:rank-1, 0:rank-1)
     *
     * [R11,R12] = [ T11, 0 ] * Y */
    if (*rank < n) {
        /* RZ factorization of [R11 R12]:
         * tau stored in work[mn..2*mn-1], sub-workspace in work[2*mn..] */
        stzrzf(*rank, n, A, lda, &work[mn], &work[2 * mn],
               lwork - 2 * mn, &iinfo);
    }

    /* B(0:m-1, 0:nrhs-1) := Q^T * B(0:m-1, 0:nrhs-1)
     * sub-workspace in work[2*mn..] */
    sormqr("L", "T", m, nrhs, mn, A, lda,
           work, B, ldb, &work[2 * mn], lwork - 2 * mn, &iinfo);

    /* B(0:rank-1, 0:nrhs-1) := inv(T11) * B(0:rank-1, 0:nrhs-1) */
    cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, *rank, nrhs, 1.0f, A, lda, B, ldb);

    /* B(rank:n-1, 0:nrhs-1) = 0 */
    for (INT j = 0; j < nrhs; j++) {
        for (INT i = *rank; i < n; i++) {
            B[i + j * ldb] = 0.0f;
        }
    }

    /* B(0:n-1, 0:nrhs-1) := Z^T * B(0:n-1, 0:nrhs-1)
     * sub-workspace in work[2*mn..] */
    if (*rank < n) {
        sormrz("L", "T", n, nrhs, *rank, n - *rank, A, lda,
               &work[mn], B, ldb, &work[2 * mn], lwork - 2 * mn, &iinfo);
    }

    /* B(0:n-1, 0:nrhs-1) := P * B(0:n-1, 0:nrhs-1)
     * Apply column permutation: result[jpvt[i]] = B[i] */
    for (INT j = 0; j < nrhs; j++) {
        for (INT i = 0; i < n; i++) {
            work[jpvt[i]] = B[i + j * ldb];
        }
        cblas_scopy(n, work, 1, &B[j * ldb], 1);
    }

    /* Undo scaling */
    if (iascl == 1) {
        slascl("G", 0, 0, anrm, smlnum, n, nrhs, B, ldb, &iinfo);
        slascl("U", 0, 0, smlnum, anrm, *rank, *rank, A, lda, &iinfo);
    } else if (iascl == 2) {
        slascl("G", 0, 0, anrm, bignum, n, nrhs, B, ldb, &iinfo);
        slascl("U", 0, 0, bignum, anrm, *rank, *rank, A, lda, &iinfo);
    }
    if (ibscl == 1) {
        slascl("G", 0, 0, smlnum, bnrm, n, nrhs, B, ldb, &iinfo);
    } else if (ibscl == 2) {
        slascl("G", 0, 0, bignum, bnrm, n, nrhs, B, ldb, &iinfo);
    }

    work[0] = (f32)lwkopt;
}
