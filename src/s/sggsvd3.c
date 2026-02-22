/**
 * @file sggsvd3.c
 * @brief SGGSVD3 computes the generalized singular value decomposition.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGGSVD3 computes the generalized singular value decomposition (GSVD)
 * of an M-by-N real matrix A and P-by-N real matrix B:
 *
 *       U**T*A*Q = D1*( 0 R ),    V**T*B*Q = D2*( 0 R )
 *
 * where U, V and Q are orthogonal matrices.
 *
 * @param[in]     jobu    = 'U': Orthogonal matrix U is computed;
 *                        = 'N': U is not computed.
 * @param[in]     jobv    = 'V': Orthogonal matrix V is computed;
 *                        = 'N': V is not computed.
 * @param[in]     jobq    = 'Q': Orthogonal matrix Q is computed;
 *                        = 'N': Q is not computed.
 * @param[in]     m       The number of rows of matrix A. m >= 0.
 * @param[in]     n       The number of columns of A and B. n >= 0.
 * @param[in]     p       The number of rows of matrix B. p >= 0.
 * @param[out]    k       Subblock dimension.
 * @param[out]    l       Subblock dimension.
 * @param[in,out] A       On entry, the M-by-N matrix A.
 *                        On exit, contains the triangular matrix R.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] B       On entry, the P-by-N matrix B.
 *                        On exit, may contain part of R.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1,p).
 * @param[out]    alpha   Generalized singular values (dimension n).
 * @param[out]    beta    Generalized singular values (dimension n).
 * @param[out]    U       Orthogonal matrix U (dimension ldu,m).
 * @param[in]     ldu     Leading dimension of U.
 * @param[out]    V       Orthogonal matrix V (dimension ldv,p).
 * @param[in]     ldv     Leading dimension of V.
 * @param[out]    Q       Orthogonal matrix Q (dimension ldq,n).
 * @param[in]     ldq     Leading dimension of Q.
 * @param[out]    work    Workspace of dimension lwork.
 * @param[in]     lwork   Dimension of work. If lwork = -1, workspace query.
 * @param[out]    iwork   Integer workspace of dimension n. Stores sorting info.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: the Jacobi-type procedure failed to converge.
 */
void sggsvd3(const char* jobu, const char* jobv, const char* jobq,
             const INT m, const INT n, const INT p,
             INT* k, INT* l,
             f32* restrict A, const INT lda,
             f32* restrict B, const INT ldb,
             f32* restrict alpha, f32* restrict beta,
             f32* restrict U, const INT ldu,
             f32* restrict V, const INT ldv,
             f32* restrict Q, const INT ldq,
             f32* restrict work, const INT lwork,
             INT* restrict iwork, INT* info)
{
    INT wantu, wantv, wantq, lquery;
    INT i, j, ibnd, isub, ncycle, lwkopt;
    f32 anorm, bnorm, smax, temp, tola = 0.0f, tolb = 0.0f, ulp, unfl;

    wantu = (jobu[0] == 'U' || jobu[0] == 'u');
    wantv = (jobv[0] == 'V' || jobv[0] == 'v');
    wantq = (jobq[0] == 'Q' || jobq[0] == 'q');
    lquery = (lwork == -1);
    lwkopt = 1;

    *info = 0;
    if (!wantu && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!wantv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -2;
    } else if (!wantq && !(jobq[0] == 'N' || jobq[0] == 'n')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -10;
    } else if (ldb < (1 > p ? 1 : p)) {
        *info = -12;
    } else if (ldu < 1 || (wantu && ldu < m)) {
        *info = -16;
    } else if (ldv < 1 || (wantv && ldv < p)) {
        *info = -18;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -20;
    } else if (lwork < 1 && !lquery) {
        *info = -22;
    }

    if (*info == 0) {
        sggsvp3(jobu, jobv, jobq, m, p, n, A, lda, B, ldb,
                tola, tolb, k, l, U, ldu, V, ldv, Q, ldq,
                iwork, NULL, work, -1, info);
        lwkopt = n + (INT)work[0];
        if (2 * n > lwkopt) lwkopt = 2 * n;
        if (lwkopt < 1) lwkopt = 1;
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SGGSVD3", -(*info));
        return;
    }
    if (lquery) {
        return;
    }

    anorm = slange("1", m, n, A, lda, work);
    bnorm = slange("1", p, n, B, ldb, work);

    ulp = slamch("P");
    unfl = slamch("S");
    tola = (m > n ? m : n) * (anorm > unfl ? anorm : unfl) * ulp;
    tolb = (p > n ? p : n) * (bnorm > unfl ? bnorm : unfl) * ulp;

    sggsvp3(jobu, jobv, jobq, m, p, n, A, lda, B, ldb,
            tola, tolb, k, l, U, ldu, V, ldv, Q, ldq,
            iwork, work, &work[n], lwork - n, info);

    stgsja(jobu, jobv, jobq, m, p, n, *k, *l, A, lda, B, ldb,
           tola, tolb, alpha, beta, U, ldu, V, ldv, Q, ldq,
           work, &ncycle, info);

    cblas_scopy(n, alpha, 1, work, 1);
    ibnd = *l;
    if (m - *k < ibnd) ibnd = m - *k;
    for (i = 0; i < ibnd; i++) {
        isub = i;
        smax = work[*k + i];
        for (j = i + 1; j < ibnd; j++) {
            temp = work[*k + j];
            if (temp > smax) {
                isub = j;
                smax = temp;
            }
        }
        if (isub != i) {
            work[*k + isub] = work[*k + i];
            work[*k + i] = smax;
            iwork[*k + i] = *k + isub;
        } else {
            iwork[*k + i] = *k + i;
        }
    }

    work[0] = (f32)lwkopt;
}
