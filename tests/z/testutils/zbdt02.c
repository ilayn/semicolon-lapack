/**
 * @file zbdt02.c
 * @brief ZBDT02 tests the change of basis C = U**H * B by computing
 *        the residual.
 *
 * Port of LAPACK's TESTING/EIG/zbdt02.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZBDT02 tests the change of basis C = U**H * B by computing the
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
 * @param[in]     C      The m by n matrix C, assumed to contain U**H * B,
 *                       dimension (ldc, n).
 * @param[in]     ldc    Leading dimension of C. ldc >= max(1, m).
 * @param[in]     U      The m by m orthogonal matrix U, dimension (ldu, m).
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, m).
 * @param[out]    work   Complex workspace array, dimension (m).
 * @param[out]    rwork  Real workspace array, dimension (m).
 * @param[out]    resid  The test ratio.
 */
void zbdt02(const INT m, const INT n,
            const c128* const restrict B, const INT ldb,
            const c128* const restrict C, const INT ldc,
            const c128* const restrict U, const INT ldu,
            c128* const restrict work, f64* const restrict rwork,
            f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT j;
    f64 bnorm, eps, realmn;

    /* Quick return if possible */
    *resid = ZERO;
    if (m <= 0 || n <= 0)
        return;
    realmn = (f64)(m > n ? m : n);
    eps = dlamch("P");

    /* Compute norm(B - U * C) */
    for (j = 0; j < n; j++) {
        cblas_zcopy(m, &B[j * ldb], 1, work, 1);
        cblas_zgemv(CblasColMajor, CblasNoTrans, m, m, &CNEGONE, U, ldu,
                    &C[j * ldc], 1, &CONE, work, 1);
        f64 colsum = cblas_dzasum(m, work, 1);
        if (colsum > *resid)
            *resid = colsum;
    }

    /* Compute norm of B. */
    bnorm = zlange("1", m, n, B, ldb, rwork);

    if (bnorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        if (bnorm >= *resid) {
            *resid = (*resid / bnorm) / (realmn * eps);
        } else {
            if (bnorm < ONE) {
                f64 tmp = fmin(*resid, realmn * bnorm);
                *resid = (tmp / bnorm) / (realmn * eps);
            } else {
                f64 tmp = fmin(*resid / bnorm, realmn);
                *resid = tmp / (realmn * eps);
            }
        }
    }
}
