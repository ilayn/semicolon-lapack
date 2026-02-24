/**
 * @file dstt21.c
 * @brief DSTT21 checks a decomposition of the form A = U S U'
 *        where A is symmetric tridiagonal, U is orthogonal,
 *        and S is diagonal (KBAND=0) or symmetric tridiagonal (KBAND=1).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * DSTT21 checks a decomposition of the form
 *
 *    A = U S U'
 *
 * where ' means transpose, A is symmetric tridiagonal, U is orthogonal,
 * and S is diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
 * Two tests are performed:
 *
 *    RESULT[0] = | A - U S U' | / ( |A| n ulp )
 *    RESULT[1] = | I - U U' | / ( n ulp )
 *
 * @param[in]     n      The size of the matrix. If zero, does nothing.
 * @param[in]     kband  The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     AD     Double array (n). Diagonal of original A.
 * @param[in]     AE     Double array (n-1). Off-diagonal of original A.
 * @param[in]     SD     Double array (n). Diagonal of S (eigenvalues if kband=0).
 * @param[in]     SE     Double array (n-1). Off-diagonal of S (if kband=1).
 * @param[in]     U      Double array (ldu, n). The orthogonal matrix U.
 * @param[in]     ldu    Leading dimension of U. ldu >= n.
 * @param[out]    work   Double array (n*(n+1)). Workspace.
 * @param[out]    result Double array (2). The test ratios.
 */
void dstt21(const INT n, const INT kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const f64* const restrict U, const INT ldu,
            f64* const restrict work, f64* restrict result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    f64 unfl = dlamch("S");
    f64 ulp = dlamch("P");

    /* ----------------------------------------------------------------
     * Test 1: RESULT[0] = | A - U S U' | / ( |A| n ulp )
     *
     * Build A in dense form in work[0..n*n-1], stored column-major with
     * leading dimension n. Only lower triangle is stored.
     * ---------------------------------------------------------------- */
    dlaset("F", n, n, ZERO, ZERO, work, n);

    /* Compute 1-norm of A as we fill it. */
    f64 anorm = ZERO;
    f64 temp1 = ZERO;
    for (INT j = 0; j < n - 1; j++) {
        /* work[(n+1)*j] is the diagonal A[j,j] in column-major */
        work[(n + 1) * j] = AD[j];
        work[(n + 1) * j + 1] = AE[j];
        f64 temp2 = fabs(AE[j]);
        anorm = fmax(anorm, fabs(AD[j]) + temp1 + temp2);
        temp1 = temp2;
    }
    work[n * n - 1] = AD[n - 1];
    anorm = fmax(anorm, fabs(AD[n - 1]) + temp1);
    anorm = fmax(anorm, unfl);

    /* Subtract U * diag(SD) * U' from work (lower triangle).
     * DSYR: work -= SD[j] * U[:,j] * U[:,j]' */
    for (INT j = 0; j < n; j++) {
        cblas_dsyr(CblasColMajor, CblasLower, n, -SD[j],
                   &U[0 + j * ldu], 1, work, n);
    }

    /* If KBAND=1, subtract off-diagonal contribution:
     * DSYR2: work -= SE[j] * (U[:,j] * U[:,j+1]' + U[:,j+1] * U[:,j]') */
    if (n > 1 && kband == 1) {
        for (INT j = 0; j < n - 1; j++) {
            cblas_dsyr2(CblasColMajor, CblasLower, n, -SE[j],
                        &U[0 + j * ldu], 1, &U[0 + (j + 1) * ldu], 1,
                        work, n);
        }
    }

    /* Compute || A - U S U' || using symmetric 1-norm (lower triangle). */
    f64 wnorm = dlansy("1", "L", n, work, n, &work[n * n]);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f64 tmp = fmin(wnorm, (f64)n * anorm);
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f64 tmp = fmin(wnorm / anorm, (f64)n);
            result[0] = tmp / (n * ulp);
        }
    }

    /* ----------------------------------------------------------------
     * Test 2: RESULT[1] = | I - U U' | / ( n ulp )
     *
     * Compute U*U' in work, then subtract I from diagonal.
     * ---------------------------------------------------------------- */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

    for (INT j = 0; j < n; j++) {
        work[(n + 1) * j] -= ONE;
    }

    f64 tmp = dlange("1", n, n, work, n, &work[n * n]);
    result[1] = fmin((f64)n, tmp) / (n * ulp);
}
