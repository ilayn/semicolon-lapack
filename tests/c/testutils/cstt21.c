/**
 * @file cstt21.c
 * @brief CSTT21 checks a decomposition of the form A = U S U**H
 *        where A is real symmetric tridiagonal, U is unitary,
 *        and S is diagonal (KBAND=0) or symmetric tridiagonal (KBAND=1).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cstt21(const INT n, const INT kband,
            const f32* const restrict AD, const f32* const restrict AE,
            const f32* const restrict SD, const f32* const restrict SE,
            const c64* const restrict U, const INT ldu,
            c64* const restrict work, f32* const restrict rwork,
            f32* restrict result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    f32 unfl = slamch("S");
    f32 ulp = slamch("P");

    claset("F", n, n, CZERO, CZERO, work, n);

    f32 anorm = ZERO;
    f32 temp1 = ZERO;
    for (INT j = 0; j < n - 1; j++) {
        work[(n + 1) * j] = CMPLXF(AD[j], 0.0f);
        work[(n + 1) * j + 1] = CMPLXF(AE[j], 0.0f);
        f32 temp2 = fabsf(AE[j]);
        anorm = fmaxf(anorm, fabsf(AD[j]) + temp1 + temp2);
        temp1 = temp2;
    }
    work[n * n - 1] = CMPLXF(AD[n - 1], 0.0f);
    anorm = fmaxf(anorm, fabsf(AD[n - 1]) + temp1);
    anorm = fmaxf(anorm, unfl);

    for (INT j = 0; j < n; j++) {
        cblas_cher(CblasColMajor, CblasLower, n, -SD[j],
                   &U[0 + j * ldu], 1, work, n);
    }

    if (n > 1 && kband == 1) {
        for (INT j = 0; j < n - 1; j++) {
            c64 alpha = CMPLXF(-SE[j], 0.0f);
            cblas_cher2(CblasColMajor, CblasLower, n, &alpha,
                        &U[0 + j * ldu], 1, &U[0 + (j + 1) * ldu], 1,
                        work, n);
        }
    }

    f32 wnorm = clanhe("1", "L", n, work, n, rwork);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f32 tmp = fminf(wnorm, (f32)n * anorm);
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f32 tmp = fminf(wnorm / anorm, (f32)n);
            result[0] = tmp / (n * ulp);
        }
    }

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

    for (INT j = 0; j < n; j++) {
        work[(n + 1) * j] = work[(n + 1) * j] - CONE;
    }

    f32 tmp = clange("1", n, n, work, n, rwork);
    result[1] = fminf((f32)n, tmp) / (n * ulp);
}
