/**
 * @file dspt01.c
 * @brief DSPT01 reconstructs a symmetric indefinite packed matrix from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include "semicolon_lapack_double.h"
#include "verify.h"
#include <math.h>

/**
 * DSPT01 reconstructs a symmetric indefinite packed matrix A from its
 * block L*D*L' or U*D*U' factorization and computes the residual
 *      norm( C - A ) / ( N * norm(A) * EPS ),
 * where C is the reconstructed matrix and EPS is the machine epsilon.
 *
 * @param[in] uplo   'U': Upper triangular, 'L': Lower triangular
 * @param[in] n      The order of the matrix A. n >= 0.
 * @param[in] A      The original symmetric matrix A, stored as a packed
 *                   triangular matrix, dimension (n*(n+1)/2).
 * @param[in] AFAC   The factored form of the matrix A, stored as a packed
 *                   triangular matrix, dimension (n*(n+1)/2).
 *                   Contains the block diagonal matrix D and the multipliers
 *                   used to obtain the factor L or U from DSPTRF.
 * @param[in] ipiv   The pivot indices from DSPTRF, dimension (n).
 * @param[out] C     Workspace array, dimension (ldc, n).
 * @param[in] ldc    The leading dimension of C. ldc >= max(1, n).
 * @param[out] rwork Workspace array, dimension (n).
 * @param[out] resid The computed residual.
 */
void dspt01(const char* uplo, const int n, const f64* A,
            const f64* AFAC, const int* ipiv, f64* C, const int ldc,
            f64* rwork, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i, j, jc;
    f64 anorm, eps;
    int info;

    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = dlansp("1", uplo, n, A, rwork);

    dlaset("F", n, n, ZERO, ONE, C, ldc);

    dlavsp(uplo, "T", "N", n, n, AFAC, ipiv, C, ldc, &info);

    dlavsp(uplo, "N", "U", n, n, AFAC, ipiv, C, ldc, &info);

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        jc = 0;
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i];
            }
            jc = jc + j + 1;
        }
    } else {
        jc = 0;
        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                C[i + j * ldc] = C[i + j * ldc] - A[jc + i - j];
            }
            jc = jc + n - j;
        }
    }

    *resid = dlansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
