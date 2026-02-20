/**
 * @file dsyt01_3.c
 * @brief DSYT01_3 reconstructs a symmetric indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization (DSYTRF_RK/DSYTRF_BK) and computes the residual.
 *
 * Port of LAPACK TESTING/LIN/dsyt01_3.f to C.
 */

#include <float.h>
#include "verify.h"

/* Forward declarations for LAPACK routines not in verify.h */
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* const restrict A, const int lda);
extern void dsyconvf_rook(const char* uplo, const char* way, const int n,
                          f64* const restrict A, const int lda,
                          f64* const restrict E, int* const restrict ipiv,
                          int* info);

/**
 * DSYT01_3 reconstructs a symmetric indefinite matrix A from its
 * block L*D*L' or U*D*U' factorization computed by DSYTRF_RK
 * (or DSYTRF_BK) and computes the residual
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
 * @param[in,out] AFAC    Diagonal of the block diagonal matrix D and factors U or L
 *                        as computed by DSYTRF_RK and DSYTRF_BK.
 *                        Double precision array, dimension (ldafac, n).
 *                        Modified during the computation (converted and reverted).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= max(1, n).
 * @param[in,out] E       On entry, contains the superdiagonal (or subdiagonal)
 *                        elements of the symmetric block diagonal matrix D.
 *                        Double precision array, dimension (n).
 *                        Modified during the computation.
 * @param[in,out] ipiv    The pivot indices from DSYTRF_RK (or DSYTRF_BK).
 *                        Integer array, dimension (n). 0-based indexing.
 *                        Modified during the computation.
 * @param[out]    C       Workspace for reconstructed matrix.
 *                        Double precision array, dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   If UPLO = 'L', norm(L*D*L' - A) / (N * norm(A) * EPS)
 *                        If UPLO = 'U', norm(U*D*U' - A) / (N * norm(A) * EPS)
 */
void dsyt01_3(
    const char* uplo,
    const int n,
    const f64* const restrict A,
    const int lda,
    f64* const restrict AFAC,
    const int ldafac,
    f64* const restrict E,
    int* const restrict ipiv,
    f64* const restrict C,
    const int ldc,
    f64* const restrict rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i, j, info;
    f64 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* a) Revert to multipliers of L */
    dsyconvf_rook(uplo, "R", n, AFAC, ldafac, E, ipiv, &info);

    /* 1) Determine EPS and the norm of A. */
    eps = DBL_EPSILON;
    anorm = dlansy("1", uplo, n, A, lda, rwork);

    /* 2) Initialize C to the identity matrix. */
    dlaset("F", n, n, ZERO, ONE, C, ldc);

    /* 3) Call DLAVSY_ROOK to form the product D * U' (or D * L'). */
    dlavsy_rook(uplo, "T", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* 4) Call DLAVSY_ROOK again to multiply by U (or L). */
    dlavsy_rook(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

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
    *resid = dlansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }

    /* b) Convert to factor of L (or U) */
    dsyconvf_rook(uplo, "C", n, AFAC, ldafac, E, ipiv, &info);
}
