/**
 * @file cspt01.c
 * @brief CSPT01 reconstructs a symmetric indefinite packed matrix A from its
 *        diagonal pivoting factorization and computes the residual.
 */

#include <math.h>
#include "verify.h"

void cspt01(const char* uplo, const INT n, const c64* A,
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
    anorm = clansp("1", uplo, n, A, rwork);

    claset("F", n, n, CZERO, CONE, C, ldc);

    clavsp(uplo, "T", "N", n, n, AFAC, ipiv, C, ldc, &info);
    clavsp(uplo, "N", "U", n, n, AFAC, ipiv, C, ldc, &info);

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        jc = 0;
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i];
            }
            jc = jc + j + 1;
        }
    } else {
        jc = 0;
        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i - j];
            }
            jc = jc + n - j;
        }
    }

    *resid = clansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
