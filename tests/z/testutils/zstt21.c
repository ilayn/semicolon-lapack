/**
 * @file zstt21.c
 * @brief ZSTT21 checks a decomposition of the form A = U S U**H
 *        where A is real symmetric tridiagonal, U is unitary,
 *        and S is diagonal (KBAND=0) or symmetric tridiagonal (KBAND=1).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zstt21(const INT n, const INT kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const c128* const restrict U, const INT ldu,
            c128* const restrict work, f64* const restrict rwork,
            f64* restrict result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    f64 unfl = dlamch("S");
    f64 ulp = dlamch("P");

    zlaset("F", n, n, CZERO, CZERO, work, n);

    f64 anorm = ZERO;
    f64 temp1 = ZERO;
    for (INT j = 0; j < n - 1; j++) {
        work[(n + 1) * j] = CMPLX(AD[j], 0.0);
        work[(n + 1) * j + 1] = CMPLX(AE[j], 0.0);
        f64 temp2 = fabs(AE[j]);
        anorm = fmax(anorm, fabs(AD[j]) + temp1 + temp2);
        temp1 = temp2;
    }
    work[n * n - 1] = CMPLX(AD[n - 1], 0.0);
    anorm = fmax(anorm, fabs(AD[n - 1]) + temp1);
    anorm = fmax(anorm, unfl);

    for (INT j = 0; j < n; j++) {
        cblas_zher(CblasColMajor, CblasLower, n, -SD[j],
                   &U[0 + j * ldu], 1, work, n);
    }

    if (n > 1 && kband == 1) {
        for (INT j = 0; j < n - 1; j++) {
            c128 alpha = CMPLX(-SE[j], 0.0);
            cblas_zher2(CblasColMajor, CblasLower, n, &alpha,
                        &U[0 + j * ldu], 1, &U[0 + (j + 1) * ldu], 1,
                        work, n);
        }
    }

    f64 wnorm = zlanhe("1", "L", n, work, n, rwork);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f64 tmp = fmin(wnorm, (f64)n * anorm);
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f64 tmp = fmin(wnorm / anorm, (f64)n);
            result[0] = tmp / (n * ulp);
        }
    }

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

    for (INT j = 0; j < n; j++) {
        work[(n + 1) * j] = work[(n + 1) * j] - CONE;
    }

    f64 tmp = zlange("1", n, n, work, n, rwork);
    result[1] = fmin((f64)n, tmp) / (n * ulp);
}
