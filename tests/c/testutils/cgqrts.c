/**
 * @file cgqrts.c
 * @brief CGQRTS tests CGGQRF, which computes the GQR factorization of an
 *        N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGQRTS tests CGGQRF, which computes the GQR factorization of an
 * N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 *
 * @param[in]     n       The number of rows of A and B.
 * @param[in]     m       The number of columns of A.
 * @param[in]     p       The number of columns of B.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, overwritten by CGGQRF.
 * @param[out]    Q       The N-by-N unitary matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of A, AF, Q, R.
 * @param[out]    taua    Scalar factors of reflectors from QR part.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, overwritten by CGGQRF.
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
void cgqrts(
    const INT n,
    const INT m,
    const INT p,
    const c64* A,
    c64* AF,
    c64* Q,
    c64* R,
    const INT lda,
    c64* taua,
    const c64* B,
    c64* BF,
    c64* Z,
    c64* T,
    c64* BWK,
    const INT ldb,
    c64* taub,
    c64* work,
    const INT lwork,
    f32* rwork,
    f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);
    const c64 CROGUE = CMPLXF(-1.0e+10f, 0.0f);
    INT info;
    f32 anorm, bnorm, resid, ulp, unfl;
    INT mn, mp;

    ulp = slamch("P");
    unfl = slamch("S");

    /* Copy the matrix A to the array AF. */

    clacpy("F", n, m, A, lda, AF, lda);
    clacpy("F", n, p, B, ldb, BF, ldb);

    anorm = clange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = clange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Factorize the matrices A and B in the arrays AF and BF. */

    cggqrf(n, m, p, AF, lda, taua, BF, ldb, taub, work, lwork, &info);

    /* Generate the N-by-N matrix Q */

    claset("F", n, n, CROGUE, CROGUE, Q, lda);
    if (n > 1)
        clacpy("L", n - 1, m, &AF[1], lda, &Q[1], lda);
    cungqr(n, n, (n < m ? n : m), Q, lda, taua, work, lwork, &info);

    /* Generate the P-by-P matrix Z */

    claset("F", p, p, CROGUE, CROGUE, Z, ldb);
    if (n <= p) {
        if (n > 0 && n < p)
            clacpy("F", n, p - n, BF, ldb, &Z[p - n], ldb);
        if (n > 1)
            clacpy("L", n - 1, n - 1, &BF[1 + (p - n) * ldb], ldb,
                   &Z[(p - n) + 1 + (p - n) * ldb], ldb);
    } else {
        if (p > 1)
            clacpy("L", p - 1, p - 1, &BF[(n - p) + 1], ldb,
                   &Z[1], ldb);
    }
    cungrq(p, p, (n < p ? n : p), Z, ldb, taub, work, lwork, &info);

    /* Copy R */

    claset("F", n, m, CZERO, CZERO, R, lda);
    clacpy("U", n, m, AF, lda, R, lda);

    /* Copy T */

    claset("F", n, p, CZERO, CZERO, T, ldb);
    if (n <= p) {
        clacpy("U", n, n, &BF[(p - n) * ldb], ldb, &T[(p - n) * ldb], ldb);
    } else {
        clacpy("F", n - p, p, BF, ldb, T, ldb);
        clacpy("U", p, p, &BF[n - p], ldb, &T[n - p], ldb);
    }

    /* Compute R - Q'*A */

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, m, n, &CNEGONE, Q, lda, A, lda, &CONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( MAX(M,N)*norm(A)*ULP ) . */

    resid = clange("1", n, m, R, lda, rwork);
    mn = 1;
    if (m > mn) mn = m;
    if (n > mn) mn = n;
    if (anorm > ZERO) {
        result[0] = ((resid / (f32)mn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    /* Compute T*Z - Q'*B */

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, p, p, &CONE, T, ldb, Z, ldb, &CZERO, BWK, ldb);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, p, n, &CNEGONE, Q, lda, B, ldb, &CONE, BWK, ldb);

    /* Compute norm( T*Z - Q'*B ) / ( MAX(P,N)*norm(B)*ULP ) . */

    resid = clange("1", n, p, BWK, ldb, rwork);
    mp = 1;
    if (p > mp) mp = p;
    if (n > mp) mp = n;
    if (bnorm > ZERO) {
        result[1] = ((resid / (f32)mp) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    /* Compute I - Q'*Q */

    claset("F", n, n, CZERO, CONE, R, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, n, -ONE, Q, lda, ONE, R, lda);

    /* Compute norm( I - Q'*Q ) / ( N * ULP ) . */

    resid = clanhe("1", "U", n, R, lda, rwork);
    result[2] = (resid / (f32)(n > 1 ? n : 1)) / ulp;

    /* Compute I - Z'*Z */

    claset("F", p, p, CZERO, CONE, T, ldb);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                p, p, -ONE, Z, ldb, ONE, T, ldb);

    /* Compute norm( I - Z'*Z ) / ( P*ULP ) . */

    resid = clanhe("1", "U", p, T, ldb, rwork);
    result[3] = (resid / (f32)(p > 1 ? p : 1)) / ulp;
}
