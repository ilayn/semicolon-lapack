/**
 * @file cget08.c
 * @brief CGET08 computes the residual for a solution of a system of linear equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cget08(
    const char* trans,
    const INT m,
    const INT n,
    const INT nrhs,
    const c64* A,
    const INT lda,
    const c64* X,
    const INT ldx,
    c64* B,
    const INT ldb,
    f32* rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT j, n1, n2;
    f32 anorm, bnorm, eps, xnorm;

    if (m <= 0 || n <= 0 || nrhs == 0) {
        *resid = ZERO;
        return;
    }

    if (trans[0] == 'T' || trans[0] == 't' || trans[0] == 'C' || trans[0] == 'c') {
        n1 = n;
        n2 = m;
    } else {
        n1 = m;
        n2 = n;
    }

    eps = slamch("E");
    anorm = clange("I", n1, n2, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'T' || trans[0] == 't') {
        cblas_trans = CblasTrans;
    } else if (trans[0] == 'C' || trans[0] == 'c') {
        cblas_trans = CblasConjTrans;
    } else {
        cblas_trans = CblasNoTrans;
    }

    cblas_cgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                n1, nrhs, n2, &CNEGONE, A, lda, X, ldx, &CONE, B, ldb);

    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        INT idx = (INT)cblas_icamax(n1, &B[j * ldb], 1);
        bnorm = cabs1f(B[idx + j * ldb]);

        idx = (INT)cblas_icamax(n2, &X[j * ldx], 1);
        xnorm = cabs1f(X[idx + j * ldx]);

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
