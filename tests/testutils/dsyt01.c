/**
 * @file dsyt01.c
 * @brief DSYT01 reconstructs a symmetric indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include <float.h>
#include "verify.h"

/* Forward declarations for LAPACK routines not in verify.h */
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* const restrict A, const int lda);

/**
 * DSYT01 reconstructs a symmetric indefinite matrix A from its
 * block L*D*L' or U*D*U' factorization and computes the residual
 *    norm( C - A ) / ( N * norm(A) * EPS ),
 * where C is the reconstructed matrix and EPS is the machine epsilon.
 *
 * @param[in]     uplo    'U': Upper triangular, 'L': Lower triangular
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     A       The original symmetric matrix.
 *                         Double precision array, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     AFAC    The factored form from DSYTRF.
 *                         Double precision array, dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= max(1, n).
 * @param[in]     ipiv    Pivot indices from DSYTRF. Integer array, dimension (n).
 *                         0-based indexing.
 * @param[out]    C       Workspace for reconstructed matrix.
 *                         Double precision array, dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   norm(C - A) / (N * norm(A) * EPS)
 */
void dsyt01(
    const char* uplo,
    const int n,
    const double* const restrict A,
    const int lda,
    const double* const restrict AFAC,
    const int ldafac,
    const int* const restrict ipiv,
    double* const restrict C,
    const int ldc,
    double* const restrict rwork,
    double* resid)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j, info;
    double anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    eps = DBL_EPSILON;
    anorm = dlansy("1", uplo, n, A, lda, rwork);

    /* Initialize C to the identity matrix. */
    dlaset("F", n, n, ZERO, ONE, C, ldc);

    /* Call DLAVSY to form the product D * U' (or D * L'). */
    dlavsy(uplo, "T", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* Call DLAVSY again to multiply by U (or L). */
    dlavsy(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* Compute the difference C - A. */
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                C[i + j * ldc] -= A[i + j * lda];
            }
        }
    }

    /* Compute norm(C - A) / (N * norm(A) * EPS). */
    *resid = dlansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (double)n) / anorm) / eps;
    }
}
