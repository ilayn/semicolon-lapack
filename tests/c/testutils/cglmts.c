/**
 * @file cglmts.c
 * @brief CGLMTS tests CGGGLM - a subroutine for solving the generalized
 *        linear model problem.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGLMTS tests CGGGLM - a subroutine for solving the generalized
 * linear model problem.
 *
 * @param[in]     n       The number of rows of the matrices A and B. n >= 0.
 * @param[in]     m       The number of columns of the matrix A. m >= 0.
 * @param[in]     p       The number of columns of the matrix B. p >= 0.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, destroyed by CGGGLM.
 * @param[in]     lda     The leading dimension of A and AF.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, destroyed by CGGGLM.
 * @param[in]     ldb     The leading dimension of B and BF.
 * @param[in]     D       The N-vector D (right-hand side).
 * @param[out]    DF      Copy of D, used as workspace.
 * @param[out]    X       The M-vector solution X.
 * @param[out]    U       The P-vector solution U.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Workspace for clange.
 * @param[out]    result  The test ratio:
 *                        norm( d - A*x - B*u ) / ((norm(A)+norm(B))*(norm(x)+norm(u))*EPS)
 */
void cglmts(
    const INT n,
    const INT m,
    const INT p,
    const c64* A,
    c64* AF,
    const INT lda,
    const c64* B,
    c64* BF,
    const INT ldb,
    const c64* D,
    c64* DF,
    c64* X,
    c64* U,
    c64* work,
    const INT lwork,
    f32* rwork,
    f32* result)
{
    const f32 ZERO = 0.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);
    INT info;
    const INT MAXEXP = FLT_MAX_EXP - 2;
    INT iscale, j;
    f32 ascl, bscl, dscl, scl, t, tnrm;
    f32 anorm, bnorm, dnorm, eps, unfl, xnorm, ynorm;

    eps = slamch("E");
    unfl = slamch("S");
    anorm = clange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = clange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Copy the matrices A and B to the arrays AF and BF,
       and the vector D the array DF. */

    clacpy("F", n, m, A, lda, AF, lda);
    clacpy("F", n, p, B, ldb, BF, ldb);
    cblas_ccopy(n, D, 1, DF, 1);

    /* Solve GLM problem */

    cggglm(n, m, p, AF, lda, BF, ldb, DF, X, U, work, lwork, &info);

    /*                       norm( d - A*x - B*u )
     *       RESULT = -----------------------------------------
     *                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
     */

    cblas_ccopy(n, D, 1, DF, 1);
    cblas_cgemv(CblasColMajor, CblasNoTrans, n, m, &CNEGONE, A, lda, X, 1, &CONE, DF, 1);

    cblas_cgemv(CblasColMajor, CblasNoTrans, n, p, &CNEGONE, B, ldb, U, 1, &CONE, DF, 1);

    dnorm = cblas_scasum(n, DF, 1);
    xnorm = cblas_scasum(m, X, 1) + cblas_scasum(p, U, 1);
    ynorm = anorm + bnorm;

    if (xnorm <= ZERO) {
        *result = ZERO;
    } else {
        *result = ((dnorm / ynorm) / xnorm) / eps;
    }

    /* A, B and d may be scaled independently: for factors a, b, d,
     * the solutions become (d/a)*x and (d/b)*u.  Test large and
     * tiny inputs, then undo solution scaling before the residual. */
    tnrm = clange("M", n, m, A, lda, rwork);
    t = clange("M", n, p, B, ldb, rwork);
    if (t > tnrm) tnrm = t;
    t = clange("M", n, 1, D, n, rwork);
    if (t > tnrm) tnrm = t;
    if (tnrm > 0.0f && tnrm <= slamch("O")) {
        /* Cases: common large, common tiny, tiny d, tiny (A,d),
         * and tiny (B,d). */
        for (iscale = 1; iscale <= 5; iscale++) {
            if (iscale == 1) {
                scl = ldexpf(1.0f, MAXEXP - (ilogbf(tnrm) + 1));
            } else {
                scl = slamch("S") / slamch("P");
                scl = ldexpf(1.0f, (ilogbf(scl) + 1) - 4 - (ilogbf(tnrm) + 1));
            }
            ascl = scl;
            bscl = scl;
            dscl = scl;
            if (iscale == 3 || iscale == 5) ascl = 1.0f;
            if (iscale == 3 || iscale == 4) bscl = 1.0f;
            clacpy("F", n, m, A, lda, AF, lda);
            clacpy("F", n, p, B, ldb, BF, ldb);
            cblas_ccopy(n, D, 1, DF, 1);
            for (j = 0; j < m; j++) {
                cblas_csscal(n, ascl, &AF[j * lda], 1);
            }
            for (j = 0; j < p; j++) {
                cblas_csscal(n, bscl, &BF[j * ldb], 1);
            }
            cblas_csscal(n, dscl, DF, 1);

            cggglm(n, m, p, AF, lda, BF, ldb, DF, X, U, work, lwork, &info);
            if (info != 0) {
                *result = 1.0f / eps;
                continue;
            }
            cblas_csscal(m, ascl / dscl, X, 1);
            cblas_csscal(p, bscl / dscl, U, 1);

            cblas_ccopy(n, D, 1, DF, 1);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n, m, &CNEGONE, A, lda, X, 1, &CONE, DF, 1);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n, p, &CNEGONE, B, ldb, U, 1, &CONE, DF, 1);
            dnorm = cblas_scasum(n, DF, 1);
            xnorm = cblas_scasum(m, X, 1) + cblas_scasum(p, U, 1);
            if (isnan(dnorm) || isnan(xnorm)) {
                *result = 1.0f / eps;
            } else if (xnorm > 0.0f) {
                t = ((dnorm / ynorm) / xnorm) / eps;
                if (t > *result) *result = t;
            }
        }
    }
}
