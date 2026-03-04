/**
 * @file zhpt01.c
 * @brief ZHPT01 reconstructs a Hermitian indefinite packed matrix A from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include <math.h>
#include "verify.h"

void zhpt01(const char* uplo, const INT n, const c128* A,
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
    anorm = zlanhp("1", uplo, n, A, rwork);

    /* Check the imaginary parts of the diagonal elements and return with
     * an error code if any are nonzero. */
    jc = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            if (cimag(AFAC[jc]) != ZERO) {
                *resid = ONE / eps;
                return;
            }
            jc = jc + j + 2;
        }
    } else {
        for (j = 0; j < n; j++) {
            if (cimag(AFAC[jc]) != ZERO) {
                *resid = ONE / eps;
                return;
            }
            jc = jc + n - j;
        }
    }

    zlaset("F", n, n, CZERO, CONE, C, ldc);

    zlavhp(uplo, "C", "N", n, n, AFAC, ipiv, C, ldc, &info);
    zlavhp(uplo, "N", "U", n, n, AFAC, ipiv, C, ldc, &info);

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        jc = 0;
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i];
            }
            C[j + j * ldc] = C[j + j * ldc] - creal(A[jc + j]);
            jc = jc + j + 1;
        }
    } else {
        jc = 0;
        for (j = 0; j < n; j++) {
            C[j + j * ldc] = C[j + j * ldc] - creal(A[jc]);
            for (i = j + 1; i < n; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i - j];
            }
            jc = jc + n - j;
        }
    }

    *resid = zlanhe("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
