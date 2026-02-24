/**
 * @file dget10.c
 * @brief DGET10 compares two matrices A and B and computes
 *        the ratio norm(A - B) / ( norm(A) * M * EPS ).
 *
 * Port of LAPACK's TESTING/EIG/dget10.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * DGET10 compares two matrices A and B and computes the ratio
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
void dget10(const INT m, const INT n,
            const f64* const restrict A, const INT lda,
            const f64* const restrict B, const INT ldb,
            f64* const restrict work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT j;
    f64 anorm, eps, unfl, wnorm;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        *result = ZERO;
        return;
    }

    unfl = dlamch("S");
    eps = dlamch("P");

    wnorm = ZERO;
    for (j = 0; j < n; j++) {
        cblas_dcopy(m, &A[j * lda], 1, work, 1);
        cblas_daxpy(m, -ONE, &B[j * ldb], 1, work, 1);
        wnorm = fmax(wnorm, cblas_dasum(n, work, 1));
    }

    anorm = fmax(dlange("1", m, n, A, lda, work), unfl);

    if (anorm > wnorm) {
        *result = (wnorm / anorm) / ((f64)m * eps);
    } else {
        if (anorm < ONE) {
            *result = (fmin(wnorm, (f64)m * anorm) / anorm) / ((f64)m * eps);
        } else {
            *result = fmin(wnorm / anorm, (f64)m) / ((f64)m * eps);
        }
    }
}
