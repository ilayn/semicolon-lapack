/**
 * @file zptt01.c
 * @brief ZPTT01 verifies the L*D*L' factorization of a Hermitian positive
 *        definite tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/LIN/zptt01.f to C.
 */

#include "semicolon_lapack_complex_double.h"
#include "verify.h"
#include <math.h>

/**
 * ZPTT01 reconstructs a tridiagonal matrix A from its L*D*L'
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
void zptt01(
    const INT n,
    const f64* const restrict D,
    const c128* const restrict E,
    const f64* const restrict DF,
    const c128* const restrict EF,
    c128* const restrict work,
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

    /* Construct the difference L*D*L' - A. */
    work[0] = CMPLX(DF[0] - D[0], 0.0);
    for (INT i = 0; i < n - 1; i++) {
        c128 de = DF[i] * EF[i];
        work[n + i] = de - E[i];
        work[i + 1] = CMPLX(creal(de * conj(EF[i])) + DF[i + 1] - D[i + 1], 0.0);
    }

    /* Compute the 1-norms of the tridiagonal matrices A and WORK. */
    f64 anorm, residval;
    if (n == 1) {
        anorm = D[0];
        residval = cabs(work[0]);
    } else {
        anorm = fmax(D[0] + cabs(E[0]), D[n - 1] + cabs(E[n - 2]));
        residval = fmax(cabs(work[0]) + cabs(work[n]),
                       cabs(work[n - 1]) + cabs(work[2 * n - 2]));
        for (INT i = 1; i < n - 1; i++) {
            anorm = fmax(anorm, D[i] + cabs(E[i]) + cabs(E[i - 1]));
            residval = fmax(residval, cabs(work[i]) + cabs(work[n + i - 1]) +
                                      cabs(work[n + i]));
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
