/**
 * @file cptt01.c
 * @brief CPTT01 verifies the L*D*L' factorization of a Hermitian positive
 *        definite tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/LIN/cptt01.f to C.
 */

#include "semicolon_lapack_complex_single.h"
#include "verify.h"
#include <math.h>

/**
 * CPTT01 reconstructs a tridiagonal matrix A from its L*D*L'
 * factorization and computes the residual
 *    norm(L*D*L' - A) / ( n * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     n     The order of the matrix A.
 * @param[in]     D     Original diagonal elements (n). Real.
 * @param[in]     E     Original subdiagonal elements (n-1). Complex.
 * @param[in]     DF    Factored diagonal elements (n). Real.
 * @param[in]     EF    Factored subdiagonal elements (n-1). Complex.
 * @param[out]    work  Complex workspace array (2*n).
 * @param[out]    resid The residual norm(L*D*L' - A) / (n * norm(A) * EPS).
 */
void cptt01(
    const INT n,
    const f32* const restrict D,
    const c64* const restrict E,
    const f32* const restrict DF,
    const c64* const restrict EF,
    c64* const restrict work,
    f32* resid)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    /* Quick return if possible */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    f32 eps = slamch("E");

    /* Construct the difference L*D*L' - A. */
    work[0] = CMPLXF(DF[0] - D[0], 0.0f);
    for (INT i = 0; i < n - 1; i++) {
        c64 de = DF[i] * EF[i];
        work[n + i] = de - E[i];
        work[i + 1] = CMPLXF(crealf(de * conjf(EF[i])) + DF[i + 1] - D[i + 1], 0.0f);
    }

    /* Compute the 1-norms of the tridiagonal matrices A and WORK. */
    f32 anorm, residval;
    if (n == 1) {
        anorm = D[0];
        residval = cabsf(work[0]);
    } else {
        anorm = fmaxf(D[0] + cabsf(E[0]), D[n - 1] + cabsf(E[n - 2]));
        residval = fmaxf(cabsf(work[0]) + cabsf(work[n]),
                       cabsf(work[n - 1]) + cabsf(work[2 * n - 2]));
        for (INT i = 1; i < n - 1; i++) {
            anorm = fmaxf(anorm, D[i] + cabsf(E[i]) + cabsf(E[i - 1]));
            residval = fmaxf(residval, cabsf(work[i]) + cabsf(work[n + i - 1]) +
                                      cabsf(work[n + i]));
        }
    }

    /* Compute norm(L*D*L' - A) / (n * norm(A) * EPS) */
    if (anorm <= ZERO) {
        if (residval != ZERO) {
            *resid = ONE / eps;
        } else {
            *resid = ZERO;
        }
    } else {
        *resid = ((residval / (f32)n) / anorm) / eps;
    }
}
