/**
 * @file zbdt05.c
 * @brief ZBDT05 reconstructs a matrix from its (partial) SVD and computes
 *        the residual.
 *
 * Port of LAPACK's TESTING/EIG/zbdt05.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zbdt05(const INT m, const INT n, const c128* const restrict A, const INT lda,
            const f64* const restrict S, const INT ns,
            const c128* const restrict U, const INT ldu,
            const c128* const restrict VT, const INT ldvt,
            c128* const restrict work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CMONE = CMPLX(-1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT i, j;
    f64 anorm, eps;
    INT minmn;

    /* Quick return if possible */
    *resid = ZERO;
    minmn = (m < n) ? m : n;
    if (minmn <= 0 || ns <= 0)
        return;

    eps = dlamch("P");
    anorm = zlange("M", m, n, A, lda, NULL);

    /* Compute U' * A * V. */
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, ns, n, &CONE, A, lda, VT, ldvt, &CZERO,
                &work[ns * ns], m);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                ns, ns, m, &CMONE, U, ldu, &work[ns * ns], m, &CZERO,
                work, ns);

    /* norm(S - U' * B * V) */
    j = 0;
    for (i = 0; i < ns; i++) {
        work[j + i] = work[j + i] + CMPLX(S[i], ZERO);
        f64 colsum = cblas_dzasum(ns, &work[j], 1);
        if (colsum > *resid) {
            *resid = colsum;
        }
        j += ns;
    }

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        if (anorm >= *resid) {
            *resid = (*resid / anorm) / ((f64)n * eps);
        } else {
            if (anorm < ONE) {
                f64 tmp = fmin(*resid, (f64)n * anorm);
                *resid = (tmp / anorm) / ((f64)n * eps);
            } else {
                f64 tmp = fmin(*resid / anorm, (f64)n);
                *resid = tmp / ((f64)n * eps);
            }
        }
    }
}
