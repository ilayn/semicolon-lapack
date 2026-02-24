/**
 * @file zhet01.c
 * @brief ZHET01 reconstructs a Hermitian indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zhet01(
    const char* uplo,
    const INT n,
    const c128* const restrict A,
    const INT lda,
    const c128* const restrict AFAC,
    const INT ldafac,
    const INT* const restrict ipiv,
    c128* const restrict C,
    const INT ldc,
    f64* const restrict rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, j, info;
    f64 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    eps = dlamch("E");
    anorm = zlanhe("1", uplo, n, A, lda, rwork);

    /* Check the imaginary parts of the diagonal elements and return with
     * an error code if any are nonzero. */
    for (j = 0; j < n; j++) {
        if (cimag(AFAC[j + j * ldafac]) != ZERO) {
            *resid = ONE / eps;
            return;
        }
    }

    /* Initialize C to the identity matrix. */
    zlaset("F", n, n, CZERO, CONE, C, ldc);

    /* Call ZLAVHE to form the product D * U' (or D * L'). */
    zlavhe(uplo, "C", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* Call ZLAVHE again to multiply by U (or L). */
    zlavhe(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* Compute the difference C - A. */
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
            C[j + j * ldc] -= (f64)creal(A[j + j * lda]);
        }
    } else {
        for (j = 0; j < n; j++) {
            C[j + j * ldc] -= (f64)creal(A[j + j * lda]);
            for (i = j + 1; i < n; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
        }
    }

    /* Compute norm(C - A) / (N * norm(A) * EPS). */
    *resid = zlanhe("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
