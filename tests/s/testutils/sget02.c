/**
 * @file sget02.c
 * @brief SGET02 computes the residual for a solution of a system of linear
 *        equations.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32 * const restrict A, const int lda,
                     f32 * const restrict work);

/**
 * SGET02 computes the residual for a solution of a system of linear
 * equations op(A)*X = B:
 *    RESID = norm(B - op(A)*X) / ( norm(op(A)) * norm(X) * EPS ),
 * where op(A) = A or A**T, depending on TRANS, and EPS is the
 * machine epsilon.
 * The norm used is the 1-norm.
 *
 * @param[in]     trans   Specifies the form of the system of equations:
 *                        = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of columns of B, the matrix of right hand
 *                        sides. nrhs >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original m x n matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,m).
 * @param[in]     X       Double precision array, dimension (ldx, nrhs).
 *                        The computed solution vectors for the system of
 *                        linear equations.
 * @param[in]     ldx     The leading dimension of the array X. If trans = 'N',
 *                        ldx >= max(1,n); if trans = 'T' or 'C', ldx >= max(1,m).
 * @param[in,out] B       Double precision array, dimension (ldb, nrhs).
 *                        On entry, the right hand side vectors for the system
 *                        of linear equations.
 *                        On exit, B is overwritten with the difference B - op(A)*X.
 * @param[in]     ldb     The leading dimension of the array B. If trans = 'N',
 *                        ldb >= max(1,m); if trans = 'T' or 'C', ldb >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (m).
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(B - op(A)*X) / ( norm(op(A)) * norm(X) * EPS ).
 */
void sget02(
    const char* trans,
    const int m,
    const int n,
    const int nrhs,
    const f32 * const restrict A,
    const int lda,
    const f32 * const restrict X,
    const int ldx,
    f32 * const restrict B,
    const int ldb,
    f32 * const restrict rwork,
    f32 *resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int j, n1, n2;
    f32 anorm, bnorm, eps, xnorm;
    int notran = (trans[0] == 'N' || trans[0] == 'n');

    // Quick exit if m = 0 or n = 0 or nrhs = 0
    if (m <= 0 || n <= 0 || nrhs == 0) {
        *resid = ZERO;
        return;
    }

    if (!notran) {
        n1 = n;
        n2 = m;
    } else {
        n1 = m;
        n2 = n;
    }

    // Exit with RESID = 1/EPS if ANORM = 0
    eps = slamch("E");
    if (notran) {
        anorm = slange("1", m, n, A, lda, rwork);
    } else {
        anorm = slange("I", m, n, A, lda, rwork);
    }
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    // Compute B - op(A)*X and store in B
    CBLAS_TRANSPOSE cblas_trans = notran ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                n1, nrhs, n2, -ONE, A, lda, X, ldx, ONE, B, ldb);

    // Compute the maximum over the number of right hand sides of
    //    norm(B - op(A)*X) / ( norm(op(A)) * norm(X) * EPS )
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_sasum(n1, &B[j * ldb], 1);
        xnorm = cblas_sasum(n2, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f32 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }
}
