/**
 * @file cget03.c
 * @brief CGET03 computes the residual for a general matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cget03(
    const INT n,
    const c64* const restrict A,
    const INT lda,
    const c64* const restrict AINV,
    const INT ldainv,
    c64* const restrict work,
    const INT ldwork,
    f32* const restrict rwork,
    f32* rcond,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT i;
    f32 ainvnm, anorm, eps;

    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    anorm = clange("1", n, n, A, lda, rwork);
    ainvnm = clange("1", n, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, &CNEGONE, AINV, ldainv, A, lda, &CZERO, work, ldwork);

    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = CONE + work[i + i * ldwork];
    }

    *resid = clange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f32)n;
}
