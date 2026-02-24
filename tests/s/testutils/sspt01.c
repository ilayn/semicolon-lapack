/**
 * @file sspt01.c
 * @brief SSPT01 reconstructs a symmetric indefinite packed matrix from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include "semicolon_lapack_single.h"
#include "verify.h"
#include <math.h>

/**
 * SSPT01 reconstructs a symmetric indefinite packed matrix A from its
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
 *                   used to obtain the factor L or U from SSPTRF.
 * @param[in] ipiv   The pivot indices from SSPTRF, dimension (n).
 * @param[out] C     Workspace array, dimension (ldc, n).
 * @param[in] ldc    The leading dimension of C. ldc >= max(1, n).
 * @param[out] rwork Workspace array, dimension (n).
 * @param[out] resid The computed residual.
 */
void sspt01(const char* uplo, const INT n, const f32* A,
            const f32* AFAC, const INT* ipiv, f32* C, const INT ldc,
            f32* rwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, j, jc;
    f32 anorm, eps;
    INT info;

    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    anorm = slansp("1", uplo, n, A, rwork);

    slaset("F", n, n, ZERO, ONE, C, ldc);

    slavsp(uplo, "T", "N", n, n, AFAC, ipiv, C, ldc, &info);

    slavsp(uplo, "N", "U", n, n, AFAC, ipiv, C, ldc, &info);

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

    *resid = slansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
