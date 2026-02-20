/**
 * @file ssyt01_rook.c
 * @brief SSYT01_ROOK reconstructs a symmetric indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (Rook pivoting) and computes the residual.
 *
 * Port of LAPACK TESTING/LIN/ssyt01_rook.f to C.
 */

#include <float.h>
#include "verify.h"

/* Forward declarations for LAPACK routines not in verify.h */
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* const restrict A, const int lda);

/**
 * SSYT01_ROOK reconstructs a symmetric indefinite matrix A from its
 * block L*D*L' or U*D*U' factorization and computes the residual
 *    norm( C - A ) / ( N * norm(A) * EPS ),
 * where C is the reconstructed matrix and EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part of the
 *                        symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       The original symmetric matrix.
 *                        Double precision array, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     AFAC    The factored form of the matrix A. AFAC contains the block
 *                        diagonal matrix D and the multipliers used to obtain the
 *                        factor L or U from the block L*D*L' or U*D*U' factorization
 *                        as computed by SSYTRF_ROOK.
 *                        Double precision array, dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= max(1, n).
 * @param[in]     ipiv    The pivot indices from SSYTRF_ROOK. Integer array, dimension (n).
 *                        0-based indexing.
 * @param[out]    C       Workspace for reconstructed matrix.
 *                        Double precision array, dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   If UPLO = 'L', norm(L*D*L' - A) / (N * norm(A) * EPS)
 *                        If UPLO = 'U', norm(U*D*U' - A) / (N * norm(A) * EPS)
 */
void ssyt01_rook(
    const char* uplo,
    const int n,
    const f32* const restrict A,
    const int lda,
    const f32* const restrict AFAC,
    const int ldafac,
    const int* const restrict ipiv,
    f32* const restrict C,
    const int ldc,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int i, j, info;
    f32 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    eps = FLT_EPSILON;
    anorm = slansy("1", uplo, n, A, lda, rwork);

    /* Initialize C to the identity matrix. */
    slaset("F", n, n, ZERO, ONE, C, ldc);

    /* Call SLAVSY_ROOK to form the product D * U' (or D * L'). */
    slavsy_rook(uplo, "T", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* Call SLAVSY_ROOK again to multiply by U (or L). */
    slavsy_rook(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

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
    *resid = slansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
