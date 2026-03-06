/**
 * @file cbdt05.c
 * @brief CBDT05 reconstructs a matrix from its (partial) SVD and computes
 *        the residual.
 *
 * Port of LAPACK's TESTING/EIG/cbdt05.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cbdt05(const INT m, const INT n, const c64* const restrict A, const INT lda,
            const f32* const restrict S, const INT ns,
            const c64* const restrict U, const INT ldu,
            const c64* const restrict VT, const INT ldvt,
            c64* const restrict work, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CMONE = CMPLXF(-1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    INT i, j;
    f32 anorm, eps;
    INT minmn;

    /* Quick return if possible */
    *resid = ZERO;
    minmn = (m < n) ? m : n;
    if (minmn <= 0 || ns <= 0)
        return;

    eps = slamch("P");
    anorm = clange("M", m, n, A, lda, NULL);

    /* Compute U' * A * V. */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, ns, n, &CONE, A, lda, VT, ldvt, &CZERO,
                &work[ns * ns], m);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                ns, ns, m, &CMONE, U, ldu, &work[ns * ns], m, &CZERO,
                work, ns);

    /* norm(S - U' * B * V) */
    j = 0;
    for (i = 0; i < ns; i++) {
        work[j + i] = work[j + i] + CMPLXF(S[i], ZERO);
        f32 colsum = cblas_scasum(ns, &work[j], 1);
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
            *resid = (*resid / anorm) / ((f32)n * eps);
        } else {
            if (anorm < ONE) {
                f32 tmp = fminf(*resid, (f32)n * anorm);
                *resid = (tmp / anorm) / ((f32)n * eps);
            } else {
                f32 tmp = fminf(*resid / anorm, (f32)n);
                *resid = tmp / ((f32)n * eps);
            }
        }
    }
}
