/**
 * @file chet01_3.c
 * @brief CHET01_3 reconstructs a Hermitian indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (CHETRF_RK/ZHETRF_BK) and
 *        computes the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void chet01_3(
    const char* uplo,
    const INT n,
    const c64* const restrict A,
    const INT lda,
    c64* const restrict AFAC,
    const INT ldafac,
    c64* const restrict E,
    const INT* const restrict ipiv,
    c64* const restrict C,
    const INT ldc,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j, info;
    f32 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* a) Revert to multipliers of L */
    csyconvf_rook(uplo, "R", n, AFAC, ldafac, E, ipiv, &info);

    /* 1) Determine EPS and the norm of A. */
    eps = slamch("E");
    anorm = clanhe("1", uplo, n, A, lda, rwork);

    /* Check the imaginary parts of the diagonal elements and return with
     * an error code if any are nonzero. */
    for (j = 0; j < n; j++) {
        if (cimagf(AFAC[j + j * ldafac]) != ZERO) {
            *resid = ONE / eps;
            return;
        }
    }

    /* 2) Initialize C to the identity matrix. */
    claset("F", n, n, CZERO, CONE, C, ldc);

    /* 3) Call CLAVHE_ROOK to form the product D * U' (or D * L'). */
    clavhe_rook(uplo, "C", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 4) Call CLAVHE_ROOK again to multiply by U (or L). */
    clavhe_rook(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 5) Compute the difference C - A. */
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
            C[j + j * ldc] -= (f32)crealf(A[j + j * lda]);
        }
    } else {
        for (j = 0; j < n; j++) {
            C[j + j * ldc] -= (f32)crealf(A[j + j * lda]);
            for (i = j + 1; i < n; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
        }
    }

    /* 6) Compute norm(C - A) / (N * norm(A) * EPS). */
    *resid = clanhe("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }

    /* b) Convert to factor of L (or U) */
    csyconvf_rook(uplo, "C", n, AFAC, ldafac, E, ipiv, &info);
}
