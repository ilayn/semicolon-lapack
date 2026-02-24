/**
 * @file zget08.c
 * @brief ZGET08 computes the residual for a solution of a system of linear equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zget08(
    const char* trans,
    const INT m,
    const INT n,
    const INT nrhs,
    const c128* A,
    const INT lda,
    const c128* X,
    const INT ldx,
    c128* B,
    const INT ldb,
    f64* rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT j, n1, n2;
    f64 anorm, bnorm, eps, xnorm;

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

    eps = dlamch("E");
    anorm = zlange("I", n1, n2, A, lda, rwork);
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

    cblas_zgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                n1, nrhs, n2, &CNEGONE, A, lda, X, ldx, &CONE, B, ldb);

    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        INT idx = (INT)cblas_izamax(n1, &B[j * ldb], 1);
        bnorm = cabs1(B[idx + j * ldb]);

        idx = (INT)cblas_izamax(n2, &X[j * ldx], 1);
        xnorm = cabs1(X[idx + j * ldx]);

        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f64 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }
}
