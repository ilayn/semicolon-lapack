/**
 * @file ssyt01_aa.c
 * @brief SSYT01_AA reconstructs a symmetric indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (Aasen's method) and computes the residual.
 *
 * Port of LAPACK TESTING/LIN/ssyt01_aa.f to C.
 */

#include <float.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* Forward declarations for LAPACK routines */
/**
 * SSYT01_AA reconstructs a symmetric indefinite matrix A from its
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
 *                        as computed by SSYTRF_AA.
 *                        Double precision array, dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= max(1, n).
 * @param[in]     ipiv    The pivot indices from SSYTRF_AA. Integer array, dimension (n).
 *                        0-based indexing.
 * @param[out]    C       Workspace for reconstructed matrix.
 *                        Double precision array, dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   If UPLO = 'L', norm(L*D*L' - A) / (N * norm(A) * EPS)
 *                        If UPLO = 'U', norm(U*D*U' - A) / (N * norm(A) * EPS)
 */
void ssyt01_aa(
    const char* uplo,
    const INT n,
    const f32* const restrict A,
    const INT lda,
    const f32* const restrict AFAC,
    const INT ldafac,
    const INT* const restrict ipiv,
    f32* const restrict C,
    const INT ldc,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, j;
    f32 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    eps = FLT_EPSILON;
    anorm = slansy("1", uplo, n, A, lda, rwork);

    /* Initialize C to the tridiagonal matrix T. */
    slaset("F", n, n, ZERO, ZERO, C, ldc);

    /* Copy diagonal of AFAC to diagonal of C:
     * Fortran: SLACPY('F', 1, N, AFAC(1,1), LDAFAC+1, C(1,1), LDC+1)
     * This copies every (LDAFAC+1)-th element, i.e., the diagonal */
    for (j = 0; j < n; j++) {
        C[j + j * ldc] = AFAC[j + j * ldafac];
    }

    if (n > 1) {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Copy superdiagonal from AFAC(1,2) with stride LDAFAC+1 to C(1,2) and C(2,1)
             * Fortran: SLACPY('F', 1, N-1, AFAC(1,2), LDAFAC+1, C(1,2), LDC+1)
             *          SLACPY('F', 1, N-1, AFAC(1,2), LDAFAC+1, C(2,1), LDC+1) */
            for (j = 0; j < n - 1; j++) {
                f32 val = AFAC[j + (j + 1) * ldafac];
                C[j + (j + 1) * ldc] = val;
                C[(j + 1) + j * ldc] = val;
            }
        } else {
            /* Copy subdiagonal from AFAC(2,1) with stride LDAFAC+1 to C(1,2) and C(2,1)
             * Fortran: SLACPY('F', 1, N-1, AFAC(2,1), LDAFAC+1, C(1,2), LDC+1)
             *          SLACPY('F', 1, N-1, AFAC(2,1), LDAFAC+1, C(2,1), LDC+1) */
            for (j = 0; j < n - 1; j++) {
                f32 val = AFAC[(j + 1) + j * ldafac];
                C[j + (j + 1) * ldc] = val;
                C[(j + 1) + j * ldc] = val;
            }
        }

        /* Call DTRMM to form the product U' * D (or L * D). */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* DTRMM('Left', 'Upper', 'Transpose', 'Unit', N-1, N, ONE, AFAC(1,2), LDAFAC, C(2,1), LDC) */
            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                        n - 1, n, ONE, &AFAC[1 * ldafac], ldafac, &C[1], ldc);
        } else {
            /* DTRMM('Left', 'Lower', 'No transpose', 'Unit', N-1, N, ONE, AFAC(2,1), LDAFAC, C(2,1), LDC) */
            cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        n - 1, n, ONE, &AFAC[1], ldafac, &C[1], ldc);
        }

        /* Call DTRMM again to multiply by U (or L). */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* DTRMM('Right', 'Upper', 'No transpose', 'Unit', N, N-1, ONE, AFAC(1,2), LDAFAC, C(1,2), LDC) */
            cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasUnit,
                        n, n - 1, ONE, &AFAC[1 * ldafac], ldafac, &C[1 * ldc], ldc);
        } else {
            /* DTRMM('Right', 'Lower', 'Transpose', 'Unit', N, N-1, ONE, AFAC(2,1), LDAFAC, C(1,2), LDC) */
            cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                        n, n - 1, ONE, &AFAC[1], ldafac, &C[1 * ldc], ldc);
        }
    }

    /* Apply symmetric pivots. */
    for (j = n - 1; j >= 0; j--) {
        i = ipiv[j];
        if (i != j) {
            cblas_sswap(n, &C[j], ldc, &C[i], ldc);
        }
    }
    for (j = n - 1; j >= 0; j--) {
        i = ipiv[j];
        if (i != j) {
            cblas_sswap(n, &C[j * ldc], 1, &C[i * ldc], 1);
        }
    }

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
