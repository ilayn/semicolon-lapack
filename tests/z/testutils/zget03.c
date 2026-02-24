/**
 * @file zget03.c
 * @brief ZGET03 computes the residual for a general matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zget03(
    const INT n,
    const c128* const restrict A,
    const INT lda,
    const c128* const restrict AINV,
    const INT ldainv,
    c128* const restrict work,
    const INT ldwork,
    f64* const restrict rwork,
    f64* rcond,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT i;
    f64 ainvnm, anorm, eps;

    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = zlange("1", n, n, A, lda, rwork);
    ainvnm = zlange("1", n, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, &CNEGONE, AINV, ldainv, A, lda, &CZERO, work, ldwork);

    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = CONE + work[i + i * ldwork];
    }

    *resid = zlange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f64)n;
}
