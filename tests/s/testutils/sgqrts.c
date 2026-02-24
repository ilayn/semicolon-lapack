/**
 * @file sgqrts.c
 * @brief SGQRTS tests SGGQRF, which computes the GQR factorization of an
 *        N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SGQRTS tests SGGQRF, which computes the GQR factorization of an
 * N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 *
 * @param[in]     n       The number of rows of A and B.
 * @param[in]     m       The number of columns of A.
 * @param[in]     p       The number of columns of B.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, overwritten by SGGQRF.
 * @param[out]    Q       The N-by-N orthogonal matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of A, AF, Q, R.
 * @param[out]    taua    Scalar factors of reflectors from QR part.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, overwritten by SGGQRF.
 * @param[out]    Z       The P-by-P orthogonal matrix Z.
 * @param[out]    T       Workspace, dimension (ldb, max(p,n)).
 * @param[out]    BWK     Workspace, dimension (ldb, n).
 * @param[in]     ldb     Leading dimension of B, BF, Z, T, BWK.
 * @param[out]    taub    Scalar factors of reflectors from RQ part.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   Dimension of work, lwork >= max(n,m,p)^2.
 * @param[out]    rwork   Workspace for norms, dimension max(n,m,p).
 * @param[out]    result  4 test ratios.
 */
void sgqrts(
    const INT n,
    const INT m,
    const INT p,
    const f32* A,
    f32* AF,
    f32* Q,
    f32* R,
    const INT lda,
    f32* taua,
    const f32* B,
    f32* BF,
    f32* Z,
    f32* T,
    f32* BWK,
    const INT ldb,
    f32* taub,
    f32* work,
    const INT lwork,
    f32* rwork,
    f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 ROGUE = -1.0e+10f;
    INT info;
    f32 anorm, bnorm, resid, ulp, unfl;
    INT mn, mp;

    ulp = slamch("P");
    unfl = slamch("S");

    /* Copy the matrix A to the array AF. */

    slacpy("F", n, m, A, lda, AF, lda);
    slacpy("F", n, p, B, ldb, BF, ldb);

    anorm = slange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = slange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Factorize the matrices A and B in the arrays AF and BF. */

    sggqrf(n, m, p, AF, lda, taua, BF, ldb, taub, work, lwork, &info);

    /* Generate the N-by-N matrix Q */

    slaset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (n > 1)
        slacpy("L", n - 1, m, &AF[1], lda, &Q[1], lda);
    sorgqr(n, n, (n < m ? n : m), Q, lda, taua, work, lwork, &info);

    /* Generate the P-by-P matrix Z */

    slaset("F", p, p, ROGUE, ROGUE, Z, ldb);
    if (n <= p) {
        if (n > 0 && n < p)
            slacpy("F", n, p - n, BF, ldb, &Z[p - n], ldb);
        if (n > 1)
            slacpy("L", n - 1, n - 1, &BF[1 + (p - n) * ldb], ldb,
                   &Z[(p - n) + 1 + (p - n) * ldb], ldb);
    } else {
        if (p > 1)
            slacpy("L", p - 1, p - 1, &BF[(n - p) + 1], ldb,
                   &Z[1], ldb);
    }
    sorgrq(p, p, (n < p ? n : p), Z, ldb, taub, work, lwork, &info);

    /* Copy R */

    slaset("F", n, m, ZERO, ZERO, R, lda);
    slacpy("U", n, m, AF, lda, R, lda);

    /* Copy T */

    slaset("F", n, p, ZERO, ZERO, T, ldb);
    if (n <= p) {
        slacpy("U", n, n, &BF[(p - n) * ldb], ldb, &T[(p - n) * ldb], ldb);
    } else {
        slacpy("F", n - p, p, BF, ldb, T, ldb);
        slacpy("U", p, p, &BF[n - p], ldb, &T[n - p], ldb);
    }

    /* Compute R - Q'*A */

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, m, n, -ONE, Q, lda, A, lda, ONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( MAX(M,N)*norm(A)*ULP ) . */

    resid = slange("1", n, m, R, lda, rwork);
    mn = 1;
    if (m > mn) mn = m;
    if (n > mn) mn = n;
    if (anorm > ZERO) {
        result[0] = ((resid / (f32)mn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    /* Compute T*Z - Q'*B */

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, p, p, ONE, T, ldb, Z, ldb, ZERO, BWK, ldb);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, p, n, -ONE, Q, lda, B, ldb, ONE, BWK, ldb);

    /* Compute norm( T*Z - Q'*B ) / ( MAX(P,N)*norm(B)*ULP ) . */

    resid = slange("1", n, p, BWK, ldb, rwork);
    mp = 1;
    if (p > mp) mp = p;
    if (n > mp) mp = n;
    if (bnorm > ZERO) {
        result[1] = ((resid / (f32)mp) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    /* Compute I - Q'*Q */

    slaset("F", n, n, ZERO, ONE, R, lda);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, n, -ONE, Q, lda, ONE, R, lda);

    /* Compute norm( I - Q'*Q ) / ( N * ULP ) . */

    resid = slansy("1", "U", n, R, lda, rwork);
    result[2] = (resid / (f32)(n > 1 ? n : 1)) / ulp;

    /* Compute I - Z'*Z */

    slaset("F", p, p, ZERO, ONE, T, ldb);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                p, p, -ONE, Z, ldb, ONE, T, ldb);

    /* Compute norm( I - Z'*Z ) / ( P*ULP ) . */

    resid = slansy("1", "U", p, T, ldb, rwork);
    result[3] = (resid / (f32)(p > 1 ? p : 1)) / ulp;
}
