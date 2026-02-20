/**
 * @file sbdt03.c
 * @brief SBDT03 reconstructs a bidiagonal matrix B from its SVD
 *        and computes the residual.
 *
 * Port of LAPACK's TESTING/EIG/sbdt03.f to C.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern f32 slamch(const char* cmach);

/**
 * SBDT03 reconstructs a bidiagonal matrix B from its SVD:
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
void sbdt03(const char* uplo, const int n, const int kd,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const int ldu,
            const f32* const restrict S,
            const f32* const restrict VT, const int ldvt,
            f32* const restrict work, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int i, j;
    f32 bnorm, eps;

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
                cblas_sgemv(CblasColMajor, CblasNoTrans, n, n, -ONE, U, ldu,
                            &work[n], 1, ZERO, work, 1);
                work[j] = work[j] + D[j];
                if (j > 0) {
                    work[j - 1] = work[j - 1] + E[j - 1];
                    bnorm = fmaxf(bnorm, fabsf(D[j]) + fabsf(E[j - 1]));
                } else {
                    bnorm = fmaxf(bnorm, fabsf(D[j]));
                }
                f32 colsum = cblas_sasum(n, work, 1);
                if (colsum > *resid)
                    *resid = colsum;
            }
        } else {
            /* B is lower bidiagonal. */
            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    work[n + i] = S[i] * VT[i + j * ldvt];
                }
                cblas_sgemv(CblasColMajor, CblasNoTrans, n, n, -ONE, U, ldu,
                            &work[n], 1, ZERO, work, 1);
                work[j] = work[j] + D[j];
                if (j < n - 1) {
                    work[j + 1] = work[j + 1] + E[j];
                    bnorm = fmaxf(bnorm, fabsf(D[j]) + fabsf(E[j]));
                } else {
                    bnorm = fmaxf(bnorm, fabsf(D[j]));
                }
                f32 colsum = cblas_sasum(n, work, 1);
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
            cblas_sgemv(CblasColMajor, CblasNoTrans, n, n, -ONE, U, ldu,
                        &work[n], 1, ZERO, work, 1);
            work[j] = work[j] + D[j];
            f32 colsum = cblas_sasum(n, work, 1);
            if (colsum > *resid)
                *resid = colsum;
        }
        j = cblas_isamax(n, D, 1);
        bnorm = fabsf(D[j]);
    }

    /* Compute norm(B - U * S * V') / ( n * norm(B) * EPS ) */
    eps = slamch("P");

    if (bnorm <= ZERO) {
        if (*resid != ZERO)
            *resid = ONE / eps;
    } else {
        if (bnorm >= *resid) {
            *resid = (*resid / bnorm) / ((f32)n * eps);
        } else {
            if (bnorm < ONE) {
                f32 tmp = fminf(*resid, (f32)n * bnorm);
                *resid = (tmp / bnorm) / ((f32)n * eps);
            } else {
                f32 tmp = fminf(*resid / bnorm, (f32)n);
                *resid = tmp / ((f32)n * eps);
            }
        }
    }
}
