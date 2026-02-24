/**
 * @file sget10.c
 * @brief SGET10 compares two matrices A and B and computes
 *        the ratio norm(A - B) / ( norm(A) * M * EPS ).
 *
 * Port of LAPACK's TESTING/EIG/sget10.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * SGET10 compares two matrices A and B and computes the ratio
 *    RESULT = norm( A - B ) / ( norm(A) * M * EPS )
 *
 * @param[in]     m       The number of rows of the matrices A and B.
 * @param[in]     n       The number of columns of the matrices A and B.
 * @param[in]     A       The m by n matrix A, dimension (lda, n).
 * @param[in]     lda     Leading dimension of A. lda >= max(1, m).
 * @param[in]     B       The m by n matrix B, dimension (ldb, n).
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1, m).
 * @param[out]    work    Workspace array, dimension (m).
 * @param[out]    result  The computed ratio.
 */
void sget10(const INT m, const INT n,
            const f32* const restrict A, const INT lda,
            const f32* const restrict B, const INT ldb,
            f32* const restrict work, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT j;
    f32 anorm, eps, unfl, wnorm;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        *result = ZERO;
        return;
    }

    unfl = slamch("S");
    eps = slamch("P");

    wnorm = ZERO;
    for (j = 0; j < n; j++) {
        cblas_scopy(m, &A[j * lda], 1, work, 1);
        cblas_saxpy(m, -ONE, &B[j * ldb], 1, work, 1);
        wnorm = fmaxf(wnorm, cblas_sasum(n, work, 1));
    }

    anorm = fmaxf(slange("1", m, n, A, lda, work), unfl);

    if (anorm > wnorm) {
        *result = (wnorm / anorm) / ((f32)m * eps);
    } else {
        if (anorm < ONE) {
            *result = (fminf(wnorm, (f32)m * anorm) / anorm) / ((f32)m * eps);
        } else {
            *result = fminf(wnorm / anorm, (f32)m) / ((f32)m * eps);
        }
    }
}
