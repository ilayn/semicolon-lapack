/**
 * @file cgtt01.c
 * @brief CGTT01 reconstructs a tridiagonal matrix A from its LU factorization
 *        and computes the residual norm(L*U - A) / (norm(A) * EPS).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGTT01 reconstructs a tridiagonal matrix A from its LU factorization
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
 * @param[out] work   Complex workspace array of dimension (ldwork * n).
 * @param[in]  ldwork The leading dimension of work. ldwork >= max(1, n).
 * @param[out] resid  The scaled residual: norm(L*U - A) / (norm(A) * EPS).
 */
void cgtt01(const INT n, const c64* DL, const c64* D, const c64* DU,
            const c64* DLF, const c64* DF, const c64* DUF, const c64* DU2,
            const INT* ipiv, c64* work, const INT ldwork, f32* resid)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT i, j, ip, lastj;
    f32 anorm, eps;
    c64 li;

    /* Quick return if possible */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("Epsilon");

    /* Copy the matrix U to WORK. */
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            work[i + j * ldwork] = CMPLXF(0.0f, 0.0f);
        }
    }

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

    /* Multiply on the left by L. */
    lastj = n - 1;
    for (i = n - 2; i >= 0; i--) {
        li = DLF[i];
        /* ZAXPY: work[i+1, i:lastj] += li * work[i, i:lastj] */
        cblas_caxpy(lastj - i + 1, &li, &work[i + i * ldwork], ldwork,
                   &work[(i + 1) + i * ldwork], ldwork);

        ip = ipiv[i];
        if (ip == i) {
            lastj = (i + 2 < n) ? (i + 2) : (n - 1);
        } else {
            cblas_cswap(lastj - i + 1, &work[i + i * ldwork], ldwork,
                       &work[(i + 1) + i * ldwork], ldwork);
        }
    }

    /* Subtract the matrix A. */
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

    /* Compute the 1-norm of the tridiagonal matrix A. */
    anorm = clangt("1", n, DL, D, DU);

    /* Compute the 1-norm of WORK, which is only guaranteed to be
       upper Hessenberg. */
    f32 wnorm = ZERO;
    for (j = 0; j < n; j++) {
        f32 colsum = ZERO;
        INT imax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
        for (i = 0; i <= imax; i++) {
            colsum += cabs1f(work[i + j * ldwork]);
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
