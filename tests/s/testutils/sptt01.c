/**
 * @file sptt01.c
 * @brief SPTT01 verifies the L*D*L' factorization of a positive definite
 *        tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/LIN/sptt01.f to C.
 */

#include "semicolon_lapack_single.h"
#include "verify.h"
#include <math.h>

/**
 * SPTT01 reconstructs a tridiagonal matrix A from its L*D*L'
 * factorization and computes the residual
 *    norm(L*D*L' - A) / ( n * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     n     The order of the matrix A.
 * @param[in]     D     Original diagonal elements (n).
 * @param[in]     E     Original subdiagonal elements (n-1).
 * @param[in]     DF    Factored diagonal elements (n).
 * @param[in]     EF    Factored subdiagonal elements (n-1).
 * @param[out]    work  Workspace array (2*n).
 * @param[out]    resid The residual norm(L*D*L' - A) / (n * norm(A) * EPS).
 */
void sptt01(
    const int n,
    const f32* const restrict D,
    const f32* const restrict E,
    const f32* const restrict DF,
    const f32* const restrict EF,
    f32* const restrict work,
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

    /* Construct the difference L*D*L' - A.
     * work[0..n-1] = diagonal differences
     * work[n..2n-2] = off-diagonal differences
     */
    work[0] = DF[0] - D[0];
    for (int i = 0; i < n - 1; i++) {
        f32 de = DF[i] * EF[i];
        work[n + i] = de - E[i];
        work[i + 1] = de * EF[i] + DF[i + 1] - D[i + 1];
    }

    /* Compute the 1-norms of the tridiagonal matrices A and WORK. */
    f32 anorm, residval;
    if (n == 1) {
        anorm = D[0];
        residval = fabsf(work[0]);
    } else {
        anorm = fmaxf(D[0] + fabsf(E[0]), D[n - 1] + fabsf(E[n - 2]));
        residval = fmaxf(fabsf(work[0]) + fabsf(work[n]),
                       fabsf(work[n - 1]) + fabsf(work[2 * n - 2]));
        for (int i = 1; i < n - 1; i++) {
            anorm = fmaxf(anorm, D[i] + fabsf(E[i]) + fabsf(E[i - 1]));
            residval = fmaxf(residval, fabsf(work[i]) + fabsf(work[n + i - 1]) +
                                      fabsf(work[n + i]));
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
