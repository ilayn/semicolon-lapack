/**
 * @file dbdt05.c
 * @brief DBDT05 reconstructs a matrix from its (partial) SVD and computes
 *        the residual.
 *
 * Port of LAPACK's TESTING/EIG/dbdt05.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * DBDT05 reconstructs a matrix A from its (partial) SVD:
 *    S = U' * A * V
 * where U and V are orthogonal matrices and S is diagonal.
 *
 * The test ratio is
 *    RESID = norm( S - U' * A * V ) / ( n * norm(A) * EPS )
 * where VT = V' and EPS is the machine precision.
 *
 * @param[in]     m      The number of rows of the matrices A and U.
 * @param[in]     n      The number of columns of the matrices A and VT.
 * @param[in]     A      The m by n matrix A, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A. lda >= max(1, m).
 * @param[in]     S      The singular values from the (partial) SVD, dimension (ns).
 * @param[in]     ns     The number of singular values/vectors from the (partial) SVD.
 * @param[in]     U      The m by ns orthogonal matrix U, dimension (ldu, ns).
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, m).
 * @param[in]     VT     The ns by n orthogonal matrix V', dimension (ldvt, n).
 * @param[in]     ldvt   Leading dimension of VT. ldvt >= max(1, ns).
 * @param[out]    work   Workspace array, dimension at least (ns*ns + m*ns).
 * @param[out]    resid  The test ratio.
 */
void dbdt05(const INT m, const INT n, const f64* const restrict A, const INT lda,
            const f64* const restrict S, const INT ns,
            const f64* const restrict U, const INT ldu,
            const f64* const restrict VT, const INT ldvt,
            f64* const restrict work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j;
    f64 anorm, eps;
    INT minmn;

    /* Quick return if possible */
    *resid = ZERO;
    minmn = (m < n) ? m : n;
    if (minmn <= 0 || ns <= 0)
        return;

    eps = dlamch("P");
    anorm = dlange("M", m, n, A, lda, work);

    /*
     * Compute U' * A * V.
     *
     * First compute A * VT' = A * V in work[ns*ns : ns*ns + m*ns - 1]
     * (m by ns matrix stored column-major)
     */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, ns, n, ONE, A, lda, VT, ldvt, ZERO,
                &work[ns * ns], m);

    /*
     * Then compute -U' * (A * V) = -U' * A * V in work[0 : ns*ns - 1]
     * (ns by ns matrix stored column-major)
     */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                ns, ns, m, -ONE, U, ldu, &work[ns * ns], m, ZERO,
                work, ns);

    /*
     * Add S to the diagonal: work = S - U' * A * V
     * (Note: we computed -U' * A * V, so adding S gives S - U' * A * V)
     */
    for (i = 0; i < ns; i++) {
        work[i + i * ns] += S[i];
    }

    /* Compute norm(S - U' * A * V) as max column absolute sum */
    *resid = ZERO;
    j = 0;
    for (i = 0; i < ns; i++) {
        f64 colsum = ZERO;
        for (INT k = 0; k < ns; k++) {
            colsum += fabs(work[j + k]);
        }
        if (colsum > *resid) {
            *resid = colsum;
        }
        j += ns;
    }

    /* Compute final residual */
    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        if (anorm >= *resid) {
            *resid = (*resid / anorm) / ((f64)n * eps);
        } else {
            if (anorm < ONE) {
                f64 tmp = fmin(*resid, (f64)n * anorm);
                *resid = (tmp / anorm) / ((f64)n * eps);
            } else {
                f64 tmp = fmin(*resid / anorm, (f64)n);
                *resid = tmp / ((f64)n * eps);
            }
        }
    }
}
