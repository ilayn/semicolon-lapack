/**
 * @file dbdt03.c
 * @brief DBDT03 reconstructs a bidiagonal matrix B from its SVD
 *        and computes the residual.
 *
 * Port of LAPACK's TESTING/EIG/dbdt03.f to C.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern double dlamch(const char* cmach);

/**
 * DBDT03 reconstructs a bidiagonal matrix B from its SVD:
 *    S = U' * B * V
 * where U and V are orthogonal matrices and S is diagonal.
 *
 * The test ratio to test the singular value decomposition is
 *    RESID = norm( B - U * S * VT ) / ( n * norm(B) * EPS )
 * where VT = V' and EPS is the machine precision.
 *
 * @param[in]     uplo   'U': upper bidiagonal. 'L': lower bidiagonal.
 * @param[in]     n      The order of the matrix B.
 * @param[in]     kd     The bandwidth of the bidiagonal matrix B. If kd = 1,
 *                       B is bidiagonal. If kd = 0, B is diagonal and E is
 *                       not referenced. If kd > 1, assumed 1. If kd < 0, assumed 0.
 * @param[in]     D      Diagonal elements of B, dimension (n).
 * @param[in]     E      Off-diagonal elements of B, dimension (n-1).
 * @param[in]     U      The n by n orthogonal matrix U, dimension (ldu, n).
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, n).
 * @param[in]     S      The singular values from the SVD, dimension (n).
 * @param[in]     VT     The n by n orthogonal matrix V', dimension (ldvt, n).
 * @param[in]     ldvt   Leading dimension of VT.
 * @param[out]    work   Workspace array, dimension (2*n).
 * @param[out]    resid  The test ratio.
 */
void dbdt03(const char* uplo, const int n, const int kd,
            const double* const restrict D, const double* const restrict E,
            const double* const restrict U, const int ldu,
            const double* const restrict S,
            const double* const restrict VT, const int ldvt,
            double* const restrict work, double* resid)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j;
    double bnorm, eps;

    /* Quick return if possible */
    *resid = ZERO;
    if (n <= 0)
        return;

    /* Compute B - U * S * V' one column at a time. */
    bnorm = ZERO;
    if (kd >= 1) {
        /* B is bidiagonal. */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* B is upper bidiagonal. */
            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    work[n + i] = S[i] * VT[i + j * ldvt];
                }
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -ONE, U, ldu,
                            &work[n], 1, ZERO, work, 1);
                work[j] = work[j] + D[j];
                if (j > 0) {
                    work[j - 1] = work[j - 1] + E[j - 1];
                    bnorm = fmax(bnorm, fabs(D[j]) + fabs(E[j - 1]));
                } else {
                    bnorm = fmax(bnorm, fabs(D[j]));
                }
                double colsum = cblas_dasum(n, work, 1);
                if (colsum > *resid)
                    *resid = colsum;
            }
        } else {
            /* B is lower bidiagonal. */
            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    work[n + i] = S[i] * VT[i + j * ldvt];
                }
                cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -ONE, U, ldu,
                            &work[n], 1, ZERO, work, 1);
                work[j] = work[j] + D[j];
                if (j < n - 1) {
                    work[j + 1] = work[j + 1] + E[j];
                    bnorm = fmax(bnorm, fabs(D[j]) + fabs(E[j]));
                } else {
                    bnorm = fmax(bnorm, fabs(D[j]));
                }
                double colsum = cblas_dasum(n, work, 1);
                if (colsum > *resid)
                    *resid = colsum;
            }
        }
    } else {
        /* B is diagonal. */
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                work[n + i] = S[i] * VT[i + j * ldvt];
            }
            cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, -ONE, U, ldu,
                        &work[n], 1, ZERO, work, 1);
            work[j] = work[j] + D[j];
            double colsum = cblas_dasum(n, work, 1);
            if (colsum > *resid)
                *resid = colsum;
        }
        j = cblas_idamax(n, D, 1);
        bnorm = fabs(D[j]);
    }

    /* Compute norm(B - U * S * V') / ( n * norm(B) * EPS ) */
    eps = dlamch("P");

    if (bnorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        if (bnorm >= *resid) {
            *resid = (*resid / bnorm) / ((double)n * eps);
        } else {
            if (bnorm < ONE) {
                double tmp = fmin(*resid, (double)n * bnorm);
                *resid = (tmp / bnorm) / ((double)n * eps);
            } else {
                double tmp = fmin(*resid / bnorm, (double)n);
                *resid = tmp / ((double)n * eps);
            }
        }
    }
}
