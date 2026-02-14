/**
 * @file dptt01.c
 * @brief DPTT01 verifies the L*D*L' factorization of a positive definite
 *        tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/LIN/dptt01.f to C.
 */

#include "semicolon_lapack_double.h"
#include "verify.h"
#include <math.h>

/**
 * DPTT01 reconstructs a tridiagonal matrix A from its L*D*L'
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
void dptt01(
    const int n,
    const f64* const restrict D,
    const f64* const restrict E,
    const f64* const restrict DF,
    const f64* const restrict EF,
    f64* const restrict work,
    f64* resid)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    /* Quick return if possible */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    f64 eps = dlamch("E");

    /* Construct the difference L*D*L' - A.
     * work[0..n-1] = diagonal differences
     * work[n..2n-2] = off-diagonal differences
     */
    work[0] = DF[0] - D[0];
    for (int i = 0; i < n - 1; i++) {
        f64 de = DF[i] * EF[i];
        work[n + i] = de - E[i];
        work[i + 1] = de * EF[i] + DF[i + 1] - D[i + 1];
    }

    /* Compute the 1-norms of the tridiagonal matrices A and WORK. */
    f64 anorm, residval;
    if (n == 1) {
        anorm = D[0];
        residval = fabs(work[0]);
    } else {
        anorm = fmax(D[0] + fabs(E[0]), D[n - 1] + fabs(E[n - 2]));
        residval = fmax(fabs(work[0]) + fabs(work[n]),
                       fabs(work[n - 1]) + fabs(work[2 * n - 2]));
        for (int i = 1; i < n - 1; i++) {
            anorm = fmax(anorm, D[i] + fabs(E[i]) + fabs(E[i - 1]));
            residval = fmax(residval, fabs(work[i]) + fabs(work[n + i - 1]) +
                                      fabs(work[n + i]));
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
        *resid = ((residval / (f64)n) / anorm) / eps;
    }
}
