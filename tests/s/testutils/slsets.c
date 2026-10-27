/**
 * @file slsets.c
 * @brief SLSETS tests SGGLSE - a subroutine for solving linear equality
 *        constrained least square problem (LSE), including scaled inputs.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SLSETS tests SGGLSE - a subroutine for solving linear equality
 * constrained least square problem (LSE), including scaled inputs.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     p       The number of rows of the matrix B. p >= 0.
 * @param[in]     n       The number of columns of the matrices A and B. n >= 0.
 * @param[in]     A       The M-by-N matrix A.
 * @param[out]    AF      Copy of A, destroyed by SGGLSE.
 * @param[in]     lda     The leading dimension of A and AF.
 * @param[in]     B       The P-by-N matrix B.
 * @param[out]    BF      Copy of B, destroyed by SGGLSE.
 * @param[in]     ldb     The leading dimension of B and BF.
 * @param[in]     C       The M-vector C.
 * @param[out]    CF      Copy of C, used as workspace.
 * @param[in]     D       The P-vector D.
 * @param[out]    DF      Copy of D, used as workspace.
 * @param[out]    X       The N-vector solution X.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Workspace for sget02.
 * @param[out]    result  The test ratios:
 *                        result[0] = norm( A*x - c )/ norm(A)*norm(X)*EPS
 *                        result[1] = norm( B*x - d )/ norm(B)*norm(X)*EPS
 */
void slsets(
    const INT m,
    const INT p,
    const INT n,
    const f32* A,
    f32* AF,
    const INT lda,
    const f32* B,
    f32* BF,
    const INT ldb,
    const f32* C,
    f32* CF,
    const f32* D,
    f32* DF,
    f32* X,
    f32* work,
    const INT lwork,
    f32* rwork,
    f32* result)
{
    INT info;
    const INT MAXEXP = FLT_MAX_EXP - 2;
    INT iscale, j;
    f32 ascl, bscl, cscl, dscl, resid, scl, t, tnrm;

    /* Copy the matrices A and B to the arrays AF and BF,
       and the vectors C and D to the arrays CF and DF, */

    slacpy("F", m, n, A, lda, AF, lda);
    slacpy("F", p, n, B, ldb, BF, ldb);
    cblas_scopy(m, C, 1, CF, 1);
    cblas_scopy(p, D, 1, DF, 1);

    /* Solve LSE problem */

    sgglse(m, n, p, AF, lda, BF, ldb, CF, DF, X, work, lwork, &info);

    /* Test the residual for the solution of LSE

       Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS */

    cblas_scopy(m, C, 1, CF, 1);
    cblas_scopy(p, D, 1, DF, 1);
    sget02("N", m, n, 1, A, lda, X, n, CF, m, rwork, &result[0]);

    /* Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS */

    sget02("N", p, n, 1, B, ldb, X, n, DF, p, rwork, &result[1]);

    /* Scaling (A,c) and (B,d) independently leaves x unchanged.
     * Scaling only (c,d) scales x by the same factor.  Exercise
     * both ends of the range and check residuals at the input scale. */
    tnrm = slange("M", m, n, A, lda, rwork);
    t = slange("M", p, n, B, ldb, rwork);
    if (t > tnrm) tnrm = t;
    t = slange("M", m, 1, C, m, rwork);
    if (t > tnrm) tnrm = t;
    t = slange("M", p, 1, D, p, rwork);
    if (t > tnrm) tnrm = t;
    if (tnrm > 0.0f && tnrm <= slamch("O")) {
        /* Cases: common large, common tiny, tiny (c,d), tiny (A,c),
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
            cscl = scl;
            dscl = scl;
            if (iscale == 3 || iscale == 5) ascl = 1.0f;
            if (iscale == 3 || iscale == 4) bscl = 1.0f;
            if (iscale == 4) dscl = 1.0f;
            if (iscale == 5) cscl = 1.0f;
            slacpy("F", m, n, A, lda, AF, lda);
            slacpy("F", p, n, B, ldb, BF, ldb);
            cblas_scopy(m, C, 1, CF, 1);
            cblas_scopy(p, D, 1, DF, 1);
            for (j = 0; j < n; j++) {
                cblas_sscal(m, ascl, &AF[j * lda], 1);
                cblas_sscal(p, bscl, &BF[j * ldb], 1);
            }
            cblas_sscal(m, cscl, CF, 1);
            cblas_sscal(p, dscl, DF, 1);

            sgglse(m, n, p, AF, lda, BF, ldb, CF, DF, X, work, lwork, &info);
            if (info != 0) {
                result[0] = 1.0f / slamch("E");
                result[1] = result[0];
                continue;
            }
            if (iscale == 3) {
                cblas_sscal(n, 1.0f / scl, X, 1);
            }

            cblas_scopy(m, C, 1, CF, 1);
            cblas_scopy(p, D, 1, DF, 1);
            sget02("N", m, n, 1, A, lda, X, n, CF, m, rwork, &resid);
            if (isnan(resid)) {
                result[0] = 1.0f / slamch("E");
            } else if (resid > result[0]) {
                result[0] = resid;
            }
            sget02("N", p, n, 1, B, ldb, X, n, DF, p, rwork, &resid);
            if (isnan(resid)) {
                result[1] = 1.0f / slamch("E");
            } else if (resid > result[1]) {
                result[1] = resid;
            }
        }
    }
}
