/**
 * @file sbdt02.c
 * @brief SBDT02 tests the change of basis C = U' * B by computing
 *        the residual.
 *
 * Port of LAPACK's TESTING/EIG/sbdt02.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * SBDT02 tests the change of basis C = U' * B by computing the
 * residual
 *
 *    RESID = norm(B - U * C) / ( max(m,n) * norm(B) * EPS ),
 *
 * where B and C are M by N matrices, U is an M by M orthogonal matrix,
 * and EPS is the machine precision.
 *
 * @param[in]     m      The number of rows of the matrices B and C and
 *                       the order of the matrix Q.
 * @param[in]     n      The number of columns of the matrices B and C.
 * @param[in]     B      The m by n matrix B, dimension (ldb, n).
 * @param[in]     ldb    Leading dimension of B. ldb >= max(1, m).
 * @param[in]     C      The m by n matrix C, assumed to contain U' * B,
 *                       dimension (ldc, n).
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[in]     U      The m by m orthogonal matrix U, dimension (ldu, m).
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, m).
 * @param[out]    work   Workspace array, dimension (m).
 * @param[out]    resid  The test ratio.
 */
void sbdt02(const INT m, const INT n,
            const f32* const restrict B, const INT ldb,
            const f32* const restrict C, const INT ldc,
            const f32* const restrict U, const INT ldu,
            f32* const restrict work, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT j;
    f32 bnorm, eps, realmn;

    /* Quick return if possible */
    *resid = ZERO;
    if (m <= 0 || n <= 0)
        return;
    realmn = (f32)(m > n ? m : n);
    eps = slamch("P");

    /* Compute norm(B - U * C) */
    for (j = 0; j < n; j++) {
        cblas_scopy(m, &B[j * ldb], 1, work, 1);
        cblas_sgemv(CblasColMajor, CblasNoTrans, m, m, -ONE, U, ldu,
                    &C[j * ldc], 1, ONE, work, 1);
        f32 colsum = cblas_sasum(m, work, 1);
        if (colsum > *resid)
            *resid = colsum;
    }

    /* Compute norm of B. */
    bnorm = slange("1", m, n, B, ldb, work);

    if (bnorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        if (bnorm >= *resid) {
            *resid = (*resid / bnorm) / (realmn * eps);
        } else {
            if (bnorm < ONE) {
                f32 tmp = fminf(*resid, realmn * bnorm);
                *resid = (tmp / bnorm) / (realmn * eps);
            } else {
                f32 tmp = fminf(*resid / bnorm, realmn);
                *resid = tmp / (realmn * eps);
            }
        }
    }
}
