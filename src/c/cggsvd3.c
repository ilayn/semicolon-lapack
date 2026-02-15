/**
 * @file cggsvd3.c
 * @brief CGGSVD3 computes the generalized singular value decomposition.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGGSVD3 computes the generalized singular value decomposition (GSVD)
 * of an M-by-N complex matrix A and P-by-N complex matrix B:
 *
 *       U**H*A*Q = D1*( 0 R ),    V**H*B*Q = D2*( 0 R )
 *
 * where U, V and Q are unitary matrices.
 *
 * @param[in]     jobu    = 'U': Unitary matrix U is computed;
 *                        = 'N': U is not computed.
 * @param[in]     jobv    = 'V': Unitary matrix V is computed;
 *                        = 'N': V is not computed.
 * @param[in]     jobq    = 'Q': Unitary matrix Q is computed;
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
 * @param[out]    U       Unitary matrix U (dimension ldu,m).
 * @param[in]     ldu     Leading dimension of U.
 * @param[out]    V       Unitary matrix V (dimension ldv,p).
 * @param[in]     ldv     Leading dimension of V.
 * @param[out]    Q       Unitary matrix Q (dimension ldq,n).
 * @param[in]     ldq     Leading dimension of Q.
 * @param[out]    work    Complex workspace of dimension lwork.
 * @param[in]     lwork   Dimension of work. If lwork = -1, workspace query.
 * @param[out]    rwork   Single precision workspace of dimension 2*n.
 * @param[out]    iwork   Integer workspace of dimension n. Stores sorting info.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1: the Jacobi-type procedure failed to converge.
 */
void cggsvd3(const char* jobu, const char* jobv, const char* jobq,
             const int m, const int n, const int p,
             int* k, int* l,
             c64* restrict A, const int lda,
             c64* restrict B, const int ldb,
             f32* restrict alpha, f32* restrict beta,
             c64* restrict U, const int ldu,
             c64* restrict V, const int ldv,
             c64* restrict Q, const int ldq,
             c64* restrict work, const int lwork,
             f32* restrict rwork,
             int* restrict iwork, int* info)
{
    int wantu, wantv, wantq, lquery;
    int i, j, ibnd, isub, ncycle, lwkopt;
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
        *info = -24;
    }

    if (*info == 0) {
        cggsvp3(jobu, jobv, jobq, m, p, n, A, lda, B, ldb,
                tola, tolb, k, l, U, ldu, V, ldv, Q, ldq,
                iwork, rwork, NULL, work, -1, info);
        lwkopt = n + (int)crealf(work[0]);
        if (2 * n > lwkopt) lwkopt = 2 * n;
        if (lwkopt < 1) lwkopt = 1;
        work[0] = CMPLXF((f32)lwkopt, 0.0f);
    }

    if (*info != 0) {
        xerbla("CGGSVD3", -(*info));
        return;
    }
    if (lquery) {
        return;
    }

    anorm = clange("1", m, n, A, lda, rwork);
    bnorm = clange("1", p, n, B, ldb, rwork);

    ulp = slamch("P");
    unfl = slamch("S");
    tola = (m > n ? m : n) * (anorm > unfl ? anorm : unfl) * ulp;
    tolb = (p > n ? p : n) * (bnorm > unfl ? bnorm : unfl) * ulp;

    cggsvp3(jobu, jobv, jobq, m, p, n, A, lda, B, ldb,
            tola, tolb, k, l, U, ldu, V, ldv, Q, ldq,
            iwork, rwork, work, &work[n], lwork - n, info);

    ctgsja(jobu, jobv, jobq, m, p, n, *k, *l, A, lda, B, ldb,
           tola, tolb, alpha, beta, U, ldu, V, ldv, Q, ldq,
           work, &ncycle, info);

    cblas_scopy(n, alpha, 1, rwork, 1);
    ibnd = *l;
    if (m - *k < ibnd) ibnd = m - *k;
    for (i = 0; i < ibnd; i++) {
        isub = i;
        smax = rwork[*k + i];
        for (j = i + 1; j < ibnd; j++) {
            temp = rwork[*k + j];
            if (temp > smax) {
                isub = j;
                smax = temp;
            }
        }
        if (isub != i) {
            rwork[*k + isub] = rwork[*k + i];
            rwork[*k + i] = smax;
            iwork[*k + i] = *k + isub;
        } else {
            iwork[*k + i] = *k + i;
        }
    }

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
