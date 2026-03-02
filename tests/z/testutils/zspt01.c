/**
 * @file zspt01.c
 * @brief ZSPT01 reconstructs a symmetric indefinite packed matrix A from its
 *        diagonal pivoting factorization and computes the residual.
 */

#include <math.h>
#include "verify.h"

void zspt01(const char* uplo, const INT n, const c128* A,
            const c128* AFAC, const INT* ipiv, c128* C, const INT ldc,
            f64* rwork, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    INT i, j, jc;
    f64 anorm, eps;
    INT info;

    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = zlansp("1", uplo, n, A, rwork);

    zlaset("F", n, n, CZERO, CONE, C, ldc);

    zlavsp(uplo, "T", "N", n, n, AFAC, ipiv, C, ldc, &info);
    zlavsp(uplo, "N", "U", n, n, AFAC, ipiv, C, ldc, &info);

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

    *resid = zlansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
