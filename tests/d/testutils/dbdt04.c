/**
 * @file dbdt04.c
 * @brief DBDT04 reconstructs a bidiagonal matrix B from its (partial) SVD
 *        and computes the residual.
 *
 * Port of LAPACK's TESTING/EIG/dbdt04.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * DBDT04 reconstructs a bidiagonal matrix B from its (partial) SVD:
 *    S = U' * B * V
 * where U and V are orthogonal matrices and S is diagonal.
 *
 * The test ratio to test the singular value decomposition is
 *    RESID = norm( S - U' * B * V ) / ( n * norm(B) * EPS )
 * where VT = V' and EPS is the machine precision.
 *
 * @param[in]     uplo   'U': upper bidiagonal. 'L': lower bidiagonal.
 * @param[in]     n      The order of the matrix B.
 * @param[in]     D      Diagonal elements of B, dimension (n).
 * @param[in]     E      Off-diagonal elements of B, dimension (n-1).
 * @param[in]     S      The ns singular values from the partial SVD,
 *                       dimension (ns), sorted in decreasing order.
 * @param[in]     ns     The number of singular values/vectors from the
 *                       partial SVD.
 * @param[in]     U      The n by ns orthogonal matrix U, dimension (ldu, ns).
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, n).
 * @param[in]     VT     The n by ns orthogonal matrix V', dimension (ldvt, n).
 * @param[in]     ldvt   Leading dimension of VT.
 * @param[out]    work   Workspace array, dimension (2*n).
 * @param[out]    resid  The test ratio.
 */
void dbdt04(const char* uplo, const INT n,
            const f64* const restrict D, const f64* const restrict E,
            const f64* const restrict S, const INT ns,
            const f64* const restrict U, const INT ldu,
            const f64* const restrict VT, const INT ldvt,
            f64* const restrict work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, k;
    f64 bnorm, eps;

    /* Quick return if possible. */
    *resid = ZERO;
    if (n <= 0 || ns <= 0)
        return;

    eps = dlamch("P");

    /* Compute S - U' * B * V. */
    bnorm = ZERO;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* B is upper bidiagonal. */
        k = 0;
        for (i = 0; i < ns; i++) {
            for (j = 0; j < n - 1; j++) {
                work[k] = D[j] * VT[i + j * ldvt] + E[j] * VT[i + (j + 1) * ldvt];
                k++;
            }
            work[k] = D[n - 1] * VT[i + (n - 1) * ldvt];
            k++;
        }
        bnorm = fabs(D[0]);
        for (i = 1; i < n; i++) {
            bnorm = fmax(bnorm, fabs(D[i]) + fabs(E[i - 1]));
        }
    } else {
        /* B is lower bidiagonal. */
        k = 0;
        for (i = 0; i < ns; i++) {
            work[k] = D[0] * VT[i];
            k++;
            for (j = 0; j < n - 1; j++) {
                work[k] = E[j] * VT[i + j * ldvt] + D[j + 1] * VT[i + (j + 1) * ldvt];
                k++;
            }
        }
        bnorm = fabs(D[n - 1]);
        for (i = 0; i < n - 1; i++) {
            bnorm = fmax(bnorm, fabs(D[i]) + fabs(E[i]));
        }
    }

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                ns, ns, n, -ONE, U, ldu, work, n, ZERO, &work[n * ns], ns);

    /* norm(S - U' * B * V) */
    k = n * ns;
    for (i = 0; i < ns; i++) {
        work[k + i] = work[k + i] + S[i];
        f64 colsum = cblas_dasum(ns, &work[k], 1);
        if (colsum > *resid)
            *resid = colsum;
        k += ns;
    }

    if (bnorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        if (bnorm >= *resid) {
            *resid = (*resid / bnorm) / ((f64)n * eps);
        } else {
            if (bnorm < ONE) {
                f64 tmp = fmin(*resid, (f64)n * bnorm);
                *resid = (tmp / bnorm) / ((f64)n * eps);
            } else {
                f64 tmp = fmin(*resid / bnorm, (f64)n);
                *resid = tmp / ((f64)n * eps);
            }
        }
    }
}
