/**
 * @file chpt01.c
 * @brief CHPT01 reconstructs a Hermitian indefinite packed matrix A from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include <math.h>
#include "verify.h"

void chpt01(const char* uplo, const INT n, const c64* A,
            const c64* AFAC, const INT* ipiv, c64* C, const INT ldc,
            f32* rwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    INT i, j, jc;
    f32 anorm, eps;
    INT info;

    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    anorm = clanhp("1", uplo, n, A, rwork);

    /* Check the imaginary parts of the diagonal elements and return with
     * an error code if any are nonzero. */
    jc = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            if (cimagf(AFAC[jc]) != ZERO) {
                *resid = ONE / eps;
                return;
            }
            jc = jc + j + 2;
        }
    } else {
        for (j = 0; j < n; j++) {
            if (cimagf(AFAC[jc]) != ZERO) {
                *resid = ONE / eps;
                return;
            }
            jc = jc + n - j;
        }
    }

    claset("F", n, n, CZERO, CONE, C, ldc);

    clavhp(uplo, "C", "N", n, n, AFAC, ipiv, C, ldc, &info);
    clavhp(uplo, "N", "U", n, n, AFAC, ipiv, C, ldc, &info);

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        jc = 0;
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i];
            }
            C[j + j * ldc] = C[j + j * ldc] - crealf(A[jc + j]);
            jc = jc + j + 1;
        }
    } else {
        jc = 0;
        for (j = 0; j < n; j++) {
            C[j + j * ldc] = C[j + j * ldc] - crealf(A[jc]);
            for (i = j + 1; i < n; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i - j];
            }
            jc = jc + n - j;
        }
    }

    *resid = clanhe("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
