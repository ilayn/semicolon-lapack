/**
 * @file spot01.c
 * @brief SPOT01 reconstructs a symmetric positive definite matrix from its
 *        Cholesky factorization and computes the residual.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern f32 slamch(const char* cmach);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);

/**
 * SPOT01 reconstructs a symmetric positive definite matrix A from
 * its L*L' or U'*U factorization and computes the residual
 *    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
 *    norm( U'*U - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in,out] AFAC    Double precision array, dimension (ldafac, n).
 *                        On entry, the factor L or U from the L*L' or U'*U
 *                        factorization. Overwritten with the reconstructed
 *                        matrix, and then with the difference L*L' - A
 *                        (or U'*U - A).
 * @param[in]     ldafac  The leading dimension of the array AFAC.
 *                        ldafac >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   norm(L*L' - A) / (N * norm(A) * EPS) or
 *                        norm(U'*U - A) / (N * norm(A) * EPS)
 */
void spot01(
    const char* uplo,
    const int n,
    const f32* const restrict A,
    const int lda,
    f32* const restrict AFAC,
    const int ldafac,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    // Quick exit if n = 0
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    // Determine EPS and the norm of A
    f32 eps = slamch("E");
    f32 anorm = slansy("1", uplo, n, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        // Compute the product U'*U, overwriting U.
        // Process columns from right to left.
        for (int k = n - 1; k >= 0; k--) {
            // Compute the (k,k) element of the result.
            // AFAC(k,k) = dot(column k of AFAC, rows 0..k)
            // Fortran: T = DDOT(K, AFAC(1,K), 1, AFAC(1,K), 1)
            f32 t = cblas_sdot(k + 1, &AFAC[k * ldafac], 1,
                                  &AFAC[k * ldafac], 1);
            AFAC[k + k * ldafac] = t;

            // Compute the rest of column k.
            // Fortran: DTRMV('Upper','Transpose','Non-unit', K-1, AFAC, LDAFAC, AFAC(1,K), 1)
            if (k > 0) {
                cblas_strmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    } else {
        // Compute the product L*L', overwriting L.
        // Process columns from right to left.
        for (int k = n - 1; k >= 0; k--) {
            // Add a multiple of column k of the factor L to each of
            // columns k+1 through n-1.
            // Fortran: DSYR('Lower', N-K, ONE, AFAC(K+1,K), 1, AFAC(K+1,K+1), LDAFAC)
            if (k + 1 < n) {
                cblas_ssyr(CblasColMajor, CblasLower,
                           n - k - 1, ONE,
                           &AFAC[(k + 1) + k * ldafac], 1,
                           &AFAC[(k + 1) + (k + 1) * ldafac], ldafac);
            }

            // Scale column k by the diagonal element.
            // Fortran: T = AFAC(K,K); DSCAL(N-K+1, T, AFAC(K,K), 1)
            f32 t = AFAC[k + k * ldafac];
            cblas_sscal(n - k, t, &AFAC[k + k * ldafac], 1);
        }
    }

    // Compute the difference L*L' - A (or U'*U - A).
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                AFAC[i + j * ldafac] -= A[i + j * lda];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                AFAC[i + j * ldafac] -= A[i + j * lda];
            }
        }
    }

    // Compute norm(L*L' - A) / (N * norm(A) * EPS)
    *resid = slansy("1", uplo, n, AFAC, ldafac, rwork);
    *resid = ((*resid / (f32)n) / anorm) / eps;
}
