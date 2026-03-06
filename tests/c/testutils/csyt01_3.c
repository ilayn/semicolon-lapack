/**
 * @file csyt01_3.c
 * @brief CSYT01_3 reconstructs a complex symmetric indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (CSYTRF_RK/ZSYTRF_BK) and computes the residual.
 *
 * Port of LAPACK TESTING/LIN/csyt01_3.f to C.
 */

#include "verify.h"

/**
 * CSYT01_3 reconstructs a symmetric indefinite matrix A from its
 * block L*D*L' or U*D*U' factorization computed by CSYTRF_RK
 * (or ZSYTRF_BK) and computes the residual
 *    norm( C - A ) / ( N * norm(A) * EPS ),
 * where C is the reconstructed matrix and EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part of the
 *                        complex symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       The original complex symmetric matrix.
 *                        Complex*16 array, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] AFAC    Diagonal of the block diagonal matrix D and factors U or L
 *                        as computed by CSYTRF_RK and ZSYTRF_BK.
 *                        Complex*16 array, dimension (ldafac, n).
 *                        Modified during the computation (converted and reverted).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= max(1, n).
 * @param[in,out] E       On entry, contains the superdiagonal (or subdiagonal)
 *                        elements of the symmetric block diagonal matrix D.
 *                        Complex*16 array, dimension (n).
 *                        Modified during the computation.
 * @param[in,out] ipiv    The pivot indices from CSYTRF_RK (or ZSYTRF_BK).
 *                        Integer array, dimension (n). 0-based indexing.
 *                        Modified during the computation.
 * @param[out]    C       Workspace for reconstructed matrix.
 *                        Complex*16 array, dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   If UPLO = 'L', norm(L*D*L' - A) / (N * norm(A) * EPS)
 *                        If UPLO = 'U', norm(U*D*U' - A) / (N * norm(A) * EPS)
 */
void csyt01_3(
    const char* uplo,
    const INT n,
    const c64* const restrict A,
    const INT lda,
    c64* const restrict AFAC,
    const INT ldafac,
    c64* const restrict E,
    INT* const restrict ipiv,
    c64* const restrict C,
    const INT ldc,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j, info;
    f32 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* a) Revert to multipliers of L */
    csyconvf_rook(uplo, "R", n, AFAC, ldafac, E, ipiv, &info);

    /* 1) Determine EPS and the norm of A. */
    eps = slamch("E");
    anorm = clansy("1", uplo, n, A, lda, rwork);

    /* 2) Initialize C to the identity matrix. */
    claset("F", n, n, CZERO, CONE, C, ldc);

    /* 3) Call CLAVSY_ROOK to form the product D * U' (or D * L'). */
    clavsy_rook(uplo, "T", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 4) Call CLAVSY_ROOK again to multiply by U (or L). */
    clavsy_rook(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 5) Compute the difference C - A. */
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

    /* 6) Compute norm(C - A) / (N * norm(A) * EPS). */
    *resid = clansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }

    /* b) Convert to factor of L (or U) */
    csyconvf_rook(uplo, "C", n, AFAC, ldafac, E, ipiv, &info);
}
