/**
 * @file zsyt01.c
 * @brief ZSYT01 reconstructs a complex symmetric indefinite matrix A from its
 *        block L*D*L' or U*D*U' factorization and computes the residual.
 */

#include "verify.h"

/**
 * ZSYT01 reconstructs a complex symmetric indefinite matrix A from its
 * block L*D*L' or U*D*U' factorization and computes the residual
 *    norm( C - A ) / ( N * norm(A) * EPS ),
 * where C is the reconstructed matrix, EPS is the machine epsilon,
 * L' is the transpose of L, and U' is the transpose of U.
 *
 * @param[in]     uplo    'U': Upper triangular, 'L': Lower triangular
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     A       The original complex symmetric matrix.
 *                         Complex*16 array, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     AFAC    The factored form from ZSYTRF.
 *                         Complex*16 array, dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= max(1, n).
 * @param[in]     ipiv    Pivot indices from ZSYTRF. Integer array, dimension (n).
 *                         0-based indexing.
 * @param[out]    C       Workspace for reconstructed matrix.
 *                         Complex*16 array, dimension (ldc, n).
 * @param[in]     ldc     The leading dimension of C. ldc >= max(1, n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   norm(C - A) / (N * norm(A) * EPS)
 */
void zsyt01(
    const char* uplo,
    const INT n,
    const c128* const restrict A,
    const INT lda,
    const c128* const restrict AFAC,
    const INT ldafac,
    const INT* const restrict ipiv,
    c128* const restrict C,
    const INT ldc,
    f64* const restrict rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, j, info;
    f64 anorm, eps;

    /* Quick exit if N = 0. */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    eps = dlamch("E");
    anorm = zlansy("1", uplo, n, A, lda, rwork);

    /* Initialize C to the identity matrix. */
    zlaset("F", n, n, CZERO, CONE, C, ldc);

    /* Call ZLAVSY to form the product D * U' (or D * L'). */
    zlavsy(uplo, "T", "N", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

    /* Call ZLAVSY again to multiply by U (or L). */
    zlavsy(uplo, "N", "U", n, n, AFAC, ldafac, ipiv, C, ldc, &info);

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
    *resid = zlansy("1", uplo, n, C, ldc, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
