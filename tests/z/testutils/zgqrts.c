/**
 * @file zgqrts.c
 * @brief ZGQRTS tests ZGGQRF, which computes the GQR factorization of an
 *        N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZGQRTS tests ZGGQRF, which computes the GQR factorization of an
 * N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 *
 * @param[in]     n       The number of rows of A and B.
 * @param[in]     m       The number of columns of A.
 * @param[in]     p       The number of columns of B.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, overwritten by ZGGQRF.
 * @param[out]    Q       The N-by-N unitary matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of A, AF, Q, R.
 * @param[out]    taua    Scalar factors of reflectors from QR part.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, overwritten by ZGGQRF.
 * @param[out]    Z       The P-by-P unitary matrix Z.
 * @param[out]    T       Workspace, dimension (ldb, max(p,n)).
 * @param[out]    BWK     Workspace, dimension (ldb, n).
 * @param[in]     ldb     Leading dimension of B, BF, Z, T, BWK.
 * @param[out]    taub    Scalar factors of reflectors from RQ part.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   Dimension of work, lwork >= max(n,m,p)^2.
 * @param[out]    rwork   Workspace for norms, dimension max(n,m,p).
 * @param[out]    result  4 test ratios.
 */
void zgqrts(
    const INT n,
    const INT m,
    const INT p,
    const c128* A,
    c128* AF,
    c128* Q,
    c128* R,
    const INT lda,
    c128* taua,
    const c128* B,
    c128* BF,
    c128* Z,
    c128* T,
    c128* BWK,
    const INT ldb,
    c128* taub,
    c128* work,
    const INT lwork,
    f64* rwork,
    f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);
    const c128 CROGUE = CMPLX(-1.0e+10, 0.0);
    INT info;
    f64 anorm, bnorm, resid, ulp, unfl;
    INT mn, mp;

    ulp = dlamch("P");
    unfl = dlamch("S");

    /* Copy the matrix A to the array AF. */

    zlacpy("F", n, m, A, lda, AF, lda);
    zlacpy("F", n, p, B, ldb, BF, ldb);

    anorm = zlange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = zlange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Factorize the matrices A and B in the arrays AF and BF. */

    zggqrf(n, m, p, AF, lda, taua, BF, ldb, taub, work, lwork, &info);

    /* Generate the N-by-N matrix Q */

    zlaset("F", n, n, CROGUE, CROGUE, Q, lda);
    if (n > 1)
        zlacpy("L", n - 1, m, &AF[1], lda, &Q[1], lda);
    zungqr(n, n, (n < m ? n : m), Q, lda, taua, work, lwork, &info);

    /* Generate the P-by-P matrix Z */

    zlaset("F", p, p, CROGUE, CROGUE, Z, ldb);
    if (n <= p) {
        if (n > 0 && n < p)
            zlacpy("F", n, p - n, BF, ldb, &Z[p - n], ldb);
        if (n > 1)
            zlacpy("L", n - 1, n - 1, &BF[1 + (p - n) * ldb], ldb,
                   &Z[(p - n) + 1 + (p - n) * ldb], ldb);
    } else {
        if (p > 1)
            zlacpy("L", p - 1, p - 1, &BF[(n - p) + 1], ldb,
                   &Z[1], ldb);
    }
    zungrq(p, p, (n < p ? n : p), Z, ldb, taub, work, lwork, &info);

    /* Copy R */

    zlaset("F", n, m, CZERO, CZERO, R, lda);
    zlacpy("U", n, m, AF, lda, R, lda);

    /* Copy T */

    zlaset("F", n, p, CZERO, CZERO, T, ldb);
    if (n <= p) {
        zlacpy("U", n, n, &BF[(p - n) * ldb], ldb, &T[(p - n) * ldb], ldb);
    } else {
        zlacpy("F", n - p, p, BF, ldb, T, ldb);
        zlacpy("U", p, p, &BF[n - p], ldb, &T[n - p], ldb);
    }

    /* Compute R - Q'*A */

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, m, n, &CNEGONE, Q, lda, A, lda, &CONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( MAX(M,N)*norm(A)*ULP ) . */

    resid = zlange("1", n, m, R, lda, rwork);
    mn = 1;
    if (m > mn) mn = m;
    if (n > mn) mn = n;
    if (anorm > ZERO) {
        result[0] = ((resid / (f64)mn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    /* Compute T*Z - Q'*B */

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, p, p, &CONE, T, ldb, Z, ldb, &CZERO, BWK, ldb);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, p, n, &CNEGONE, Q, lda, B, ldb, &CONE, BWK, ldb);

    /* Compute norm( T*Z - Q'*B ) / ( MAX(P,N)*norm(B)*ULP ) . */

    resid = zlange("1", n, p, BWK, ldb, rwork);
    mp = 1;
    if (p > mp) mp = p;
    if (n > mp) mp = n;
    if (bnorm > ZERO) {
        result[1] = ((resid / (f64)mp) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    /* Compute I - Q'*Q */

    zlaset("F", n, n, CZERO, CONE, R, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, n, -ONE, Q, lda, ONE, R, lda);

    /* Compute norm( I - Q'*Q ) / ( N * ULP ) . */

    resid = zlanhe("1", "U", n, R, lda, rwork);
    result[2] = (resid / (f64)(n > 1 ? n : 1)) / ulp;

    /* Compute I - Z'*Z */

    zlaset("F", p, p, CZERO, CONE, T, ldb);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                p, p, -ONE, Z, ldb, ONE, T, ldb);

    /* Compute norm( I - Z'*Z ) / ( P*ULP ) . */

    resid = zlanhe("1", "U", p, T, ldb, rwork);
    result[3] = (resid / (f64)(p > 1 ? p : 1)) / ulp;
}
