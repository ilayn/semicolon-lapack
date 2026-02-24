/**
 * @file zhet01_3.c
 * @brief ZHET01_3 reconstructs a Hermitian indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (ZHETRF_RK/ZHETRF_BK) and
 *        computes the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zhet01_3(
    const char* uplo,
    const INT n,
    const c128* const restrict A,
    const INT lda,
    c128* const restrict AFAC,
    const INT ldafac,
    c128* const restrict E,
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

    /* a) Revert to multipliers of L */
    zsyconvf_rook(uplo, "R", n, AFAC, ldafac, E, ipiv, &info);

    /* 1) Determine EPS and the norm of A. */
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

    /* 2) Initialize C to the identity matrix. */
    zlaset("F", n, n, CZERO, CONE, C, ldc);

    /* 3) Call ZLAVHE_ROOK to form the product D * U' (or D * L'). */
    zlavhe_rook(uplo, "C", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 4) Call ZLAVHE_ROOK again to multiply by U (or L). */
    zlavhe_rook(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 5) Compute the difference C - A. */
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

    /* 6) Compute norm(C - A) / (N * norm(A) * EPS). */
    *resid = zlanhe("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }

    /* b) Convert to factor of L (or U) */
    zsyconvf_rook(uplo, "C", n, AFAC, ldafac, E, ipiv, &info);
}
