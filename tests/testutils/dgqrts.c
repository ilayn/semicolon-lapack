/**
 * @file dgqrts.c
 * @brief DGQRTS tests DGGQRF, which computes the GQR factorization of an
 *        N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* A, const int lda, double* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern void dggqrf(const int n, const int m, const int p,
                   double* A, const int lda, double* taua,
                   double* B, const int ldb, double* taub,
                   double* work, const int lwork, int* info);
extern void dorgqr(const int m, const int n, const int k,
                   double* A, const int lda, const double* tau,
                   double* work, const int lwork, int* info);
extern void dorgrq(const int m, const int n, const int k,
                   double* A, const int lda, const double* tau,
                   double* work, const int lwork, int* info);

/**
 * DGQRTS tests DGGQRF, which computes the GQR factorization of an
 * N-by-M matrix A and a N-by-P matrix B: A = Q*R and B = Q*T*Z.
 *
 * @param[in]     n       The number of rows of A and B.
 * @param[in]     m       The number of columns of A.
 * @param[in]     p       The number of columns of B.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, overwritten by DGGQRF.
 * @param[out]    Q       The N-by-N orthogonal matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of A, AF, Q, R.
 * @param[out]    taua    Scalar factors of reflectors from QR part.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, overwritten by DGGQRF.
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
void dgqrts(
    const int n,
    const int m,
    const int p,
    const double* A,
    double* AF,
    double* Q,
    double* R,
    const int lda,
    double* taua,
    const double* B,
    double* BF,
    double* Z,
    double* T,
    double* BWK,
    const int ldb,
    double* taub,
    double* work,
    const int lwork,
    double* rwork,
    double* result)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double ROGUE = -1.0e+10;
    int info;
    double anorm, bnorm, resid, ulp, unfl;
    int mn, mp;

    ulp = dlamch("P");
    unfl = dlamch("S");

    /* Copy the matrix A to the array AF. */

    dlacpy("F", n, m, A, lda, AF, lda);
    dlacpy("F", n, p, B, ldb, BF, ldb);

    anorm = dlange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = dlange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Factorize the matrices A and B in the arrays AF and BF. */

    dggqrf(n, m, p, AF, lda, taua, BF, ldb, taub, work, lwork, &info);

    /* Generate the N-by-N matrix Q */

    dlaset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (n > 1)
        dlacpy("L", n - 1, m, &AF[1], lda, &Q[1], lda);
    dorgqr(n, n, (n < m ? n : m), Q, lda, taua, work, lwork, &info);

    /* Generate the P-by-P matrix Z */

    dlaset("F", p, p, ROGUE, ROGUE, Z, ldb);
    if (n <= p) {
        if (n > 0 && n < p)
            dlacpy("F", n, p - n, BF, ldb, &Z[p - n], ldb);
        if (n > 1)
            dlacpy("L", n - 1, n - 1, &BF[1 + (p - n) * ldb], ldb,
                   &Z[(p - n) + 1 + (p - n) * ldb], ldb);
    } else {
        if (p > 1)
            dlacpy("L", p - 1, p - 1, &BF[(n - p) + 1], ldb,
                   &Z[1], ldb);
    }
    dorgrq(p, p, (n < p ? n : p), Z, ldb, taub, work, lwork, &info);

    /* Copy R */

    dlaset("F", n, m, ZERO, ZERO, R, lda);
    dlacpy("U", n, m, AF, lda, R, lda);

    /* Copy T */

    dlaset("F", n, p, ZERO, ZERO, T, ldb);
    if (n <= p) {
        dlacpy("U", n, n, &BF[(p - n) * ldb], ldb, &T[(p - n) * ldb], ldb);
    } else {
        dlacpy("F", n - p, p, BF, ldb, T, ldb);
        dlacpy("U", p, p, &BF[n - p], ldb, &T[n - p], ldb);
    }

    /* Compute R - Q'*A */

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, m, n, -ONE, Q, lda, A, lda, ONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( MAX(M,N)*norm(A)*ULP ) . */

    resid = dlange("1", n, m, R, lda, rwork);
    mn = 1;
    if (m > mn) mn = m;
    if (n > mn) mn = n;
    if (anorm > ZERO) {
        result[0] = ((resid / (double)mn) / anorm) / ulp;
    } else {
        result[0] = ZERO;
    }

    /* Compute T*Z - Q'*B */

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, p, p, ONE, T, ldb, Z, ldb, ZERO, BWK, ldb);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, p, n, -ONE, Q, lda, B, ldb, ONE, BWK, ldb);

    /* Compute norm( T*Z - Q'*B ) / ( MAX(P,N)*norm(B)*ULP ) . */

    resid = dlange("1", n, p, BWK, ldb, rwork);
    mp = 1;
    if (p > mp) mp = p;
    if (n > mp) mp = n;
    if (bnorm > ZERO) {
        result[1] = ((resid / (double)mp) / bnorm) / ulp;
    } else {
        result[1] = ZERO;
    }

    /* Compute I - Q'*Q */

    dlaset("F", n, n, ZERO, ONE, R, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, n, -ONE, Q, lda, ONE, R, lda);

    /* Compute norm( I - Q'*Q ) / ( N * ULP ) . */

    resid = dlansy("1", "U", n, R, lda, rwork);
    result[2] = (resid / (double)(n > 1 ? n : 1)) / ulp;

    /* Compute I - Z'*Z */

    dlaset("F", p, p, ZERO, ONE, T, ldb);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                p, p, -ONE, Z, ldb, ONE, T, ldb);

    /* Compute norm( I - Z'*Z ) / ( P*ULP ) . */

    resid = dlansy("1", "U", p, T, ldb, rwork);
    result[3] = (resid / (double)(p > 1 ? p : 1)) / ulp;
}
