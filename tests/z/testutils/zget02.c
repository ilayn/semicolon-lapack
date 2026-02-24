/**
 * @file zget02.c
 * @brief ZGET02 computes the residual for a solution of a system of linear
 *        equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zget02(
    const char* trans,
    const INT m,
    const INT n,
    const INT nrhs,
    const c128* const restrict A,
    const INT lda,
    const c128* const restrict X,
    const INT ldx,
    c128* const restrict B,
    const INT ldb,
    f64* const restrict rwork,
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
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = zlange("1", m, n, A, lda, rwork);
    } else {
        anorm = zlange("I", m, n, A, lda, rwork);
    }
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
        bnorm = cblas_dzasum(n1, &B[j * ldb], 1);
        xnorm = cblas_dzasum(n2, &X[j * ldx], 1);
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
