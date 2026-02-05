/**
 * @file dbdt01.c
 * @brief DBDT01 reconstructs a general matrix A from its bidiagonal form
 *        and computes the residual.
 *
 * Port of LAPACK's TESTING/EIG/dbdt01.f to C.
 */

#include <math.h>
#include <string.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);

/**
 * DBDT01 reconstructs a general matrix A from its bidiagonal form
 *    A = Q * B * P'
 * where Q (m by min(m,n)) and P' (min(m,n) by n) are orthogonal
 * matrices and B is bidiagonal.
 *
 * The test ratio is
 *    RESID = norm(A - Q * B * P') / ( n * norm(A) * EPS )
 * where EPS is the machine precision.
 *
 * @param[in]     m      The number of rows of the matrices A and Q.
 * @param[in]     n      The number of columns of the matrices A and P'.
 * @param[in]     kd     If kd = 0, B is diagonal and E is not referenced.
 *                       If kd = 1, the reduction was performed by xGEBRD; B is upper
 *                       bidiagonal if m >= n, and lower bidiagonal if m < n.
 *                       If kd = -1, the reduction was performed by xGBBRD; B is
 *                       always upper bidiagonal.
 * @param[in]     A      The m by n matrix A, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A. lda >= max(1, m).
 * @param[in]     Q      The m by min(m,n) orthogonal matrix Q, dimension (ldq, min(m,n)).
 * @param[in]     ldq    Leading dimension of Q. ldq >= max(1, m).
 * @param[in]     D      Diagonal elements of B, dimension (min(m,n)).
 * @param[in]     E      Off-diagonal elements of B, dimension (min(m,n)-1).
 * @param[in]     PT     The min(m,n) by n orthogonal matrix P', dimension (ldpt, n).
 * @param[in]     ldpt   Leading dimension of PT. ldpt >= max(1, min(m,n)).
 * @param[out]    work   Workspace array, dimension (m+n).
 * @param[out]    resid  The test ratio.
 */
void dbdt01(const int m, const int n, const int kd,
            const double* const restrict A, const int lda,
            const double* const restrict Q, const int ldq,
            const double* const restrict D, const double* const restrict E,
            const double* const restrict PT, const int ldpt,
            double* const restrict work, double* resid)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j;
    double anorm, eps;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        *resid = ZERO;
        return;
    }

    (void)(m < n);  /* mnmin computed in Fortran but unused here */

    /* Compute A - Q * B * P' one column at a time. */
    *resid = ZERO;

    if (kd != 0) {
        /* B is bidiagonal. */

        if (kd != 0 && m >= n) {
            /* B is upper bidiagonal and m >= n. */
            for (j = 0; j < n; j++) {
                /* Copy column j of A to work[0:m-1] */
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                /* Compute B * P'(j,:) in work[m:m+n-1] */
                for (i = 0; i < n - 1; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt] + E[i] * PT[i + 1 + j * ldpt];
                }
                work[m + n - 1] = D[n - 1] * PT[n - 1 + j * ldpt];

                /* Compute work = work - Q * work[m:m+n-1] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -ONE,
                            Q, ldq, &work[m], 1, ONE, work, 1);

                /* Accumulate max absolute column sum */
                double colsum = ZERO;
                for (i = 0; i < m; i++) {
                    colsum += fabs(work[i]);
                }
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        } else if (kd < 0) {
            /* B is upper bidiagonal and m < n. */
            for (j = 0; j < n; j++) {
                /* Copy column j of A to work[0:m-1] */
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                /* Compute B * P'(j,:) in work[m:m+m-1] */
                for (i = 0; i < m - 1; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt] + E[i] * PT[i + 1 + j * ldpt];
                }
                work[m + m - 1] = D[m - 1] * PT[m - 1 + j * ldpt];

                /* Compute work = work - Q * work[m:m+m-1] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, -ONE,
                            Q, ldq, &work[m], 1, ONE, work, 1);

                /* Accumulate max absolute column sum */
                double colsum = ZERO;
                for (i = 0; i < m; i++) {
                    colsum += fabs(work[i]);
                }
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        } else {
            /* B is lower bidiagonal. */
            for (j = 0; j < n; j++) {
                /* Copy column j of A to work[0:m-1] */
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                /* Compute B * P'(j,:) in work[m:m+m-1] */
                work[m] = D[0] * PT[j * ldpt];
                for (i = 1; i < m; i++) {
                    work[m + i] = E[i - 1] * PT[i - 1 + j * ldpt] + D[i] * PT[i + j * ldpt];
                }

                /* Compute work = work - Q * work[m:m+m-1] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, -ONE,
                            Q, ldq, &work[m], 1, ONE, work, 1);

                /* Accumulate max absolute column sum */
                double colsum = ZERO;
                for (i = 0; i < m; i++) {
                    colsum += fabs(work[i]);
                }
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        }
    } else {
        /* B is diagonal. */
        if (m >= n) {
            for (j = 0; j < n; j++) {
                /* Copy column j of A to work[0:m-1] */
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                /* Compute D * P'(j,:) in work[m:m+n-1] */
                for (i = 0; i < n; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt];
                }

                /* Compute work = work - Q * work[m:m+n-1] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -ONE,
                            Q, ldq, &work[m], 1, ONE, work, 1);

                /* Accumulate max absolute column sum */
                double colsum = ZERO;
                for (i = 0; i < m; i++) {
                    colsum += fabs(work[i]);
                }
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                /* Copy column j of A to work[0:m-1] */
                for (i = 0; i < m; i++) {
                    work[i] = A[i + j * lda];
                }

                /* Compute D * P'(j,:) in work[m:m+m-1] */
                for (i = 0; i < m; i++) {
                    work[m + i] = D[i] * PT[i + j * ldpt];
                }

                /* Compute work = work - Q * work[m:m+m-1] */
                cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, -ONE,
                            Q, ldq, &work[m], 1, ONE, work, 1);

                /* Accumulate max absolute column sum */
                double colsum = ZERO;
                for (i = 0; i < m; i++) {
                    colsum += fabs(work[i]);
                }
                if (colsum > *resid) {
                    *resid = colsum;
                }
            }
        }
    }

    /* Compute norm(A - Q * B * P') / ( n * norm(A) * EPS ) */
    anorm = dlange("1", m, n, A, lda, work);
    eps = dlamch("P");

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        if (anorm >= *resid) {
            *resid = (*resid / anorm) / ((double)n * eps);
        } else {
            if (anorm < ONE) {
                double tmp = fmin(*resid, (double)n * anorm);
                *resid = (tmp / anorm) / ((double)n * eps);
            } else {
                double tmp = fmin(*resid / anorm, (double)n);
                *resid = tmp / ((double)n * eps);
            }
        }
    }
}
