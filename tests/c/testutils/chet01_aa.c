/**
 * @file chet01_aa.c
 * @brief CHET01_AA reconstructs a Hermitian indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (Aasen's method) and computes
 *        the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void chet01_aa(
    const char* uplo,
    const INT n,
    const c64* const restrict A,
    const INT lda,
    const c64* const restrict AFAC,
    const INT ldafac,
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

    INT i, j;
    f32 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    eps = slamch("E");
    anorm = clanhe("1", uplo, n, A, lda, rwork);

    /* Initialize C to the tridiagonal matrix T. */
    claset("F", n, n, CZERO, CZERO, C, ldc);

    /* Copy diagonal of AFAC to diagonal of C:
     * Fortran: CLACPY('F', 1, N, AFAC(1,1), LDAFAC+1, C(1,1), LDC+1) */
    for (j = 0; j < n; j++) {
        C[j + j * ldc] = AFAC[j + j * ldafac];
    }

    if (n > 1) {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Copy superdiagonal from AFAC(0, 1) with stride LDAFAC+1 */
            for (j = 0; j < n - 1; j++) {
                c64 val = AFAC[j + (j + 1) * ldafac];
                C[j + (j + 1) * ldc] = val;
                C[(j + 1) + j * ldc] = val;
            }
            /* Conjugate the lower copy to make it Hermitian */
            clacgv(n - 1, &C[1], ldc + 1);
        } else {
            /* Copy subdiagonal from AFAC(1, 0) with stride LDAFAC+1 */
            for (j = 0; j < n - 1; j++) {
                c64 val = AFAC[(j + 1) + j * ldafac];
                C[j + (j + 1) * ldc] = val;
                C[(j + 1) + j * ldc] = val;
            }
            /* Conjugate the upper copy to make it Hermitian */
            clacgv(n - 1, &C[ldc], ldc + 1);
        }

        /* Call ZTRMM to form the product U' * D (or L * D). */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* ZTRMM('Left', 'Upper', 'Conjugate transpose', 'Unit',
             *       N-1, N, CONE, AFAC(0,1), LDAFAC, C(1,0), LDC) */
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasUnit,
                        n - 1, n, &CONE, &AFAC[1 * ldafac], ldafac, &C[1], ldc);
        } else {
            /* ZTRMM('Left', 'Lower', 'No transpose', 'Unit',
             *       N-1, N, CONE, AFAC(1,0), LDAFAC, C(1,0), LDC) */
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        n - 1, n, &CONE, &AFAC[1], ldafac, &C[1], ldc);
        }

        /* Call ZTRMM again to multiply by U (or L). */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* ZTRMM('Right', 'Upper', 'No transpose', 'Unit',
             *       N, N-1, CONE, AFAC(0,1), LDAFAC, C(0,1), LDC) */
            cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                        n, n - 1, &CONE, &AFAC[1 * ldafac], ldafac, &C[1 * ldc], ldc);
        } else {
            /* ZTRMM('Right', 'Lower', 'Conjugate transpose', 'Unit',
             *       N, N-1, CONE, AFAC(1,0), LDAFAC, C(0,1), LDC) */
            cblas_ctrmm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans, CblasUnit,
                        n, n - 1, &CONE, &AFAC[1], ldafac, &C[1 * ldc], ldc);
        }

        /* Apply hermitian pivots */
        for (j = n - 1; j >= 0; j--) {
            i = ipiv[j];
            if (i != j)
                cblas_cswap(n, &C[j], ldc, &C[i], ldc);
        }
        for (j = n - 1; j >= 0; j--) {
            i = ipiv[j];
            if (i != j)
                cblas_cswap(n, &C[j * ldc], 1, &C[i * ldc], 1);
        }
    }

    /* Compute the difference C - A. */
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
        }
    }

    /* Compute norm(C - A) / (N * norm(A) * EPS). */
    *resid = clanhe("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
