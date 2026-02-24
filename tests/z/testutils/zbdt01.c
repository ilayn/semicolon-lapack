/**
 * @file zbdt01.c
 * @brief ZBDT01 reconstructs a general matrix A from its bidiagonal form
 *        and computes the residual.
 *
 * Port of LAPACK's TESTING/EIG/zbdt01.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZBDT01 reconstructs a general matrix A from its bidiagonal form
 *    A = Q * B * P**H
 * where Q (m by min(m,n)) and P**H (min(m,n) by n) are unitary
 * matrices and B is bidiagonal.
 *
 * The test ratio is
 *    RESID = norm(A - Q * B * P**H) / ( n * norm(A) * EPS )
 * where EPS is the machine precision.
 *
 * @param[in]     m      The number of rows of the matrices A and Q.
 * @param[in]     n      The number of columns of the matrices A and P**H.
 * @param[in]     kd     If kd = 0, B is diagonal and E is not referenced.
 *                       If kd = 1, the reduction was performed by xGEBRD; B is upper
 *                       bidiagonal if m >= n, and lower bidiagonal if m < n.
 *                       If kd = -1, the reduction was performed by xGBBRD; B is
 *                       always upper bidiagonal.
 * @param[in]     A      The m by n matrix A, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A. lda >= max(1, m).
 * @param[in]     Q      The m by min(m,n) unitary matrix Q, dimension (ldq, min(m,n)).
 * @param[in]     ldq    Leading dimension of Q. ldq >= max(1, m).
 * @param[in]     D      Diagonal elements of B, dimension (min(m,n)).
 * @param[in]     E      Off-diagonal elements of B, dimension (min(m,n)-1).
 * @param[in]     PT     The min(m,n) by n unitary matrix P**H, dimension (ldpt, n).
 * @param[in]     ldpt   Leading dimension of PT. ldpt >= max(1, min(m,n)).
 * @param[out]    work   Complex workspace array, dimension (m+n).
 * @param[out]    rwork  Real workspace array, dimension (m).
 * @param[out]    resid  The test ratio.
 */
void zbdt01(const INT m, const INT n, const INT kd,
            const c128* const restrict A, const INT lda,
            const c128* const restrict Q, const INT ldq,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict PT, const INT ldpt,
            c128* const restrict work, f64* const restrict rwork,
            f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT i, j;
    f64 anorm, eps;

    if (m <= 0 || n <= 0) {
        *resid = ZERO;
        return;
    }

    *resid = ZERO;

    if (kd != 0) {

        if (kd != 0 && m >= n) {

            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                for (i = 0; i < n - 1; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt] + E[i] * PT[i + 1 + j * ldpt];
                }
                work[m + n - 1] = D[n - 1] * PT[n - 1 + j * ldpt];

                cblas_zgemv(CblasColMajor, CblasNoTrans, m, n, &CNEGONE,
                            Q, ldq, &work[m], 1, &CONE, work, 1);

                f64 colsum = cblas_dzasum(m, work, 1);
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        } else if (kd < 0) {

            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                for (i = 0; i < m - 1; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt] + E[i] * PT[i + 1 + j * ldpt];
                }
                work[m + m - 1] = D[m - 1] * PT[m - 1 + j * ldpt];

                cblas_zgemv(CblasColMajor, CblasNoTrans, m, m, &CNEGONE,
                            Q, ldq, &work[m], 1, &CONE, work, 1);

                f64 colsum = cblas_dzasum(m, work, 1);
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        } else {

            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                work[m] = D[0] * PT[j * ldpt];
                for (i = 1; i < m; i++) {
                    work[m + i] = E[i - 1] * PT[i - 1 + j * ldpt] + D[i] * PT[i + j * ldpt];
                }

                cblas_zgemv(CblasColMajor, CblasNoTrans, m, m, &CNEGONE,
                            Q, ldq, &work[m], 1, &CONE, work, 1);

                f64 colsum = cblas_dzasum(m, work, 1);
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        }
    } else {

        if (m >= n) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                for (i = 0; i < n; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt];
                }

                cblas_zgemv(CblasColMajor, CblasNoTrans, m, n, &CNEGONE,
                            Q, ldq, &work[m], 1, &CONE, work, 1);

                f64 colsum = cblas_dzasum(m, work, 1);
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                for (i = 0; i < m; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt];
                }

                cblas_zgemv(CblasColMajor, CblasNoTrans, m, m, &CNEGONE,
                            Q, ldq, &work[m], 1, &CONE, work, 1);

                f64 colsum = cblas_dzasum(m, work, 1);
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        }
    }

    anorm = zlange("1", m, n, A, lda, rwork);
    eps = dlamch("P");

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
