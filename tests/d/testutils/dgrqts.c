/**
 * @file dgrqts.c
 * @brief DGRQTS tests DGGRQF, which computes the GRQ factorization of an
 *        M-by-N matrix A and a P-by-N matrix B: A = R*Q and B = Z*T*Q.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DGRQTS tests DGGRQF, which computes the GRQ factorization of an
 * M-by-N matrix A and a P-by-N matrix B: A = R*Q and B = Z*T*Q.
 *
 * @param[in]     m       The number of rows of A.
 * @param[in]     p       The number of rows of B.
 * @param[in]     n       The number of columns of A and B.
 * @param[in]     A       The M-by-N matrix A.
 * @param[out]    AF      Copy of A, overwritten by DGGRQF.
 * @param[out]    Q       The N-by-N orthogonal matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of A, AF, Q, R.
 * @param[out]    taua    Scalar factors of reflectors from RQ part.
 * @param[in]     B       The P-by-N matrix B.
 * @param[out]    BF      Copy of B, overwritten by DGGRQF.
 * @param[out]    Z       The P-by-P orthogonal matrix Z.
 * @param[out]    T       Workspace, dimension (ldb, max(p,n)).
 * @param[out]    BWK     Workspace, dimension (ldb, n).
 * @param[in]     ldb     Leading dimension of B, BF, Z, T, BWK.
 * @param[out]    taub    Scalar factors of reflectors from QR part.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   Dimension of work, lwork >= max(m,p,n)^2.
 * @param[out]    rwork   Workspace for norms, dimension max(m,p,n).
 * @param[out]    result  4 test ratios.
 */
void dgrqts(
    const INT m,
    const INT p,
    const INT n,
    const f64* A,
    f64* AF,
    f64* Q,
    f64* R,
    const INT lda,
    f64* taua,
    const f64* B,
    f64* BF,
    f64* Z,
    f64* T,
    f64* BWK,
    const INT ldb,
    f64* taub,
    f64* work,
    const INT lwork,
    f64* rwork,
    f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 ROGUE = -1.0e+10;
    INT info;
    f64 anorm, bnorm, resid, ulp, unfl;
    INT mn, mp;

    ulp = dlamch("P");
    unfl = dlamch("S");

    /* Copy the matrix A to the array AF. */

    dlacpy("F", m, n, A, lda, AF, lda);
    dlacpy("F", p, n, B, ldb, BF, ldb);

    anorm = dlange("1", m, n, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = dlange("1", p, n, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Factorize the matrices A and B in the arrays AF and BF. */

    dggrqf(m, p, n, AF, lda, taua, BF, ldb, taub, work, lwork, &info);

    /* Generate the N-by-N matrix Q */

    dlaset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (m <= n) {
        if (m > 0 && m < n)
            dlacpy("F", m, n - m, AF, lda, &Q[n - m], lda);
        if (m > 1)
            dlacpy("L", m - 1, m - 1, &AF[1 + (n - m) * lda], lda,
                   &Q[(n - m) + 1 + (n - m) * lda], lda);
    } else {
        if (n > 1)
            dlacpy("L", n - 1, n - 1, &AF[(m - n) + 1], lda,
                   &Q[1], lda);
    }
    dorgrq(n, n, (m < n ? m : n), Q, lda, taua, work, lwork, &info);

    /* Generate the P-by-P matrix Z */

    dlaset("F", p, p, ROGUE, ROGUE, Z, ldb);
    if (p > 1)
        dlacpy("L", p - 1, n, &BF[1], ldb, &Z[1], ldb);
    dorgqr(p, p, (p < n ? p : n), Z, ldb, taub, work, lwork, &info);

    /* Copy R */

    dlaset("F", m, n, ZERO, ZERO, R, lda);
    if (m <= n) {
        dlacpy("U", m, m, &AF[(n - m) * lda], lda, &R[(n - m) * lda], lda);
    } else {
        dlacpy("F", m - n, n, AF, lda, R, lda);
        dlacpy("U", n, n, &AF[m - n], lda, &R[m - n], lda);
    }

    /* Copy T */

    dlaset("F", p, n, ZERO, ZERO, T, ldb);
    dlacpy("U", p, n, BF, ldb, T, ldb);

    /* Compute R - A*Q' */

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, n, -ONE, A, lda, Q, lda, ONE, R, lda);

    /* Compute norm( R - A*Q' ) / ( MAX(M,N)*norm(A)*ULP ) . */

    resid = dlange("1", m, n, R, lda, rwork);
    mn = 1;
    if (m > mn) mn = m;
    if (n > mn) mn = n;
    if (anorm > ZERO) {
        result[0] = ((resid / (f64)mn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    /* Compute T*Q - Z'*B */

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                p, n, p, ONE, Z, ldb, B, ldb, ZERO, BWK, ldb);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                p, n, n, ONE, T, ldb, Q, lda, -ONE, BWK, ldb);

    /* Compute norm( T*Q - Z'*B ) / ( MAX(P,M)*norm(B)*ULP ) . */

    resid = dlange("1", p, n, BWK, ldb, rwork);
    mp = 1;
    if (p > mp) mp = p;
    if (m > mp) mp = m;
    if (bnorm > ZERO) {
        result[1] = ((resid / (f64)mp) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    /* Compute I - Q*Q' */

    dlaset("F", n, n, ZERO, ONE, R, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -ONE, Q, lda, ONE, R, lda);

    /* Compute norm( I - Q*Q' ) / ( N * ULP ) . */

    resid = dlansy("1", "U", n, R, lda, rwork);
    result[2] = (resid / (f64)(n > 1 ? n : 1)) / ulp;

    /* Compute I - Z'*Z */

    dlaset("F", p, p, ZERO, ONE, T, ldb);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                p, p, -ONE, Z, ldb, ONE, T, ldb);

    /* Compute norm( I - Z'*Z ) / ( P*ULP ) . */

    resid = dlansy("1", "U", p, T, ldb, rwork);
    result[3] = (resid / (f64)(p > 1 ? p : 1)) / ulp;
}
