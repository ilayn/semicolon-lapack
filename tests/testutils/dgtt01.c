/**
 * @file dgtt01.c
 * @brief DGTT01 reconstructs a tridiagonal matrix A from its LU factorization
 *        and computes the residual norm(L*U - A) / (norm(A) * EPS).
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern double dlamch(const char* cmach);
extern double dlangt(const char* norm, const int n,
                     const double* const restrict DL,
                     const double* const restrict D,
                     const double* const restrict DU);

/**
 * DGTT01 reconstructs a tridiagonal matrix A from its LU factorization
 * and computes the residual
 *    norm(L*U - A) / (norm(A) * EPS),
 * where EPS is the machine epsilon.
 *
 * @param[in]  n      The order of the matrix A. n >= 0.
 * @param[in]  DL     The (n-1) sub-diagonal elements of A. Array of dimension (n-1).
 * @param[in]  D      The diagonal elements of A. Array of dimension (n).
 * @param[in]  DU     The (n-1) super-diagonal elements of A. Array of dimension (n-1).
 * @param[in]  DLF    The (n-1) multipliers that define L. Array of dimension (n-1).
 * @param[in]  DF     The n diagonal elements of U. Array of dimension (n).
 * @param[in]  DUF    The (n-1) elements of first super-diagonal of U. Array of dimension (n-1).
 * @param[in]  DU2    The (n-2) elements of second super-diagonal of U. Array of dimension (n-2).
 * @param[in]  ipiv   The pivot indices. Array of dimension (n).
 * @param[out] work   Workspace array of dimension (ldwork * n).
 * @param[in]  ldwork The leading dimension of work. ldwork >= max(1, n).
 * @param[out] resid  The scaled residual: norm(L*U - A) / (norm(A) * EPS).
 */
void dgtt01(
    const int n,
    const double * const restrict DL,
    const double * const restrict D,
    const double * const restrict DU,
    const double * const restrict DLF,
    const double * const restrict DF,
    const double * const restrict DUF,
    const double * const restrict DU2,
    const int * const restrict ipiv,
    double * const restrict work,
    const int ldwork,
    double *resid)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j, ip, lastj;
    double anorm, eps, li;
    double wnorm;

    /* Quick return if possible */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");

    /* Initialize work to zero and copy U to work */
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            work[i + j * ldwork] = ZERO;
        }
    }

    /* Copy U to work (0-based indexing) */
    for (i = 0; i < n; i++) {
        /* Diagonal */
        work[i + i * ldwork] = DF[i];

        /* First super-diagonal */
        if (i < n - 1) {
            work[i + (i + 1) * ldwork] = DUF[i];
        }

        /* Second super-diagonal */
        if (i < n - 2) {
            work[i + (i + 2) * ldwork] = DU2[i];
        }
    }

    /* Multiply on the left by L (working backwards) */
    lastj = n - 1;
    for (i = n - 2; i >= 0; i--) {
        li = DLF[i];
        /* DAXPY: work[i+1, i:lastj] += li * work[i, i:lastj] */
        for (j = i; j <= lastj; j++) {
            work[(i + 1) + j * ldwork] += li * work[i + j * ldwork];
        }

        ip = ipiv[i];
        if (ip == i) {
            /* No swap needed */
            lastj = (i + 2 < n) ? (i + 2) : (n - 1);
        } else {
            /* Swap rows i and i+1 in columns i:lastj */
            for (j = i; j <= lastj; j++) {
                double temp = work[i + j * ldwork];
                work[i + j * ldwork] = work[(i + 1) + j * ldwork];
                work[(i + 1) + j * ldwork] = temp;
            }
        }
    }

    /* Subtract the matrix A from L*U */
    work[0] = work[0] - D[0];
    if (n > 1) {
        work[0 + 1 * ldwork] = work[0 + 1 * ldwork] - DU[0];
        work[(n - 1) + (n - 2) * ldwork] = work[(n - 1) + (n - 2) * ldwork] - DL[n - 2];
        work[(n - 1) + (n - 1) * ldwork] = work[(n - 1) + (n - 1) * ldwork] - D[n - 1];
        for (i = 1; i < n - 1; i++) {
            work[i + (i - 1) * ldwork] = work[i + (i - 1) * ldwork] - DL[i - 1];
            work[i + i * ldwork] = work[i + i * ldwork] - D[i];
            work[i + (i + 1) * ldwork] = work[i + (i + 1) * ldwork] - DU[i];
        }
    }

    /* Compute the 1-norm of the tridiagonal matrix A */
    anorm = dlangt("1", n, DL, D, DU);

    /* Compute the 1-norm of work (upper Hessenberg) */
    /* Since the result is upper Hessenberg, compute column sums manually */
    wnorm = ZERO;
    for (j = 0; j < n; j++) {
        double colsum = ZERO;
        /* Upper Hessenberg: nonzeros in column j are rows 0 to min(j+1, n-1) */
        int imax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
        for (i = 0; i <= imax; i++) {
            colsum += fabs(work[i + j * ldwork]);
        }
        if (colsum > wnorm) {
            wnorm = colsum;
        }
    }

    /* Compute norm(L*U - A) / (norm(A) * EPS) */
    if (anorm <= ZERO) {
        if (wnorm != ZERO) {
            *resid = ONE / eps;
        } else {
            *resid = ZERO;
        }
    } else {
        *resid = (wnorm / anorm) / eps;
    }
}
