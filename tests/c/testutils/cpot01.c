/**
 * @file cpot01.c
 * @brief CPOT01 reconstructs a Hermitian positive definite matrix from its
 *        Cholesky factorization and computes the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CPOT01 reconstructs a Hermitian positive definite matrix A from
 * its L*L' or U'*U factorization and computes the residual
 *    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
 *    norm( U'*U - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon, L' is the conjugate transpose of L,
 * and U' is the conjugate transpose of U.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the Hermitian matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       Complex*16 array, dimension (lda, n).
 *                        The original Hermitian matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in,out] AFAC    Complex*16 array, dimension (ldafac, n).
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
void cpot01(
    const char* uplo,
    const INT n,
    const c64* const restrict A,
    const INT lda,
    c64* const restrict AFAC,
    const INT ldafac,
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

    // Exit with RESID = 1/EPS if ANORM = 0
    f32 eps = slamch("E");
    f32 anorm = clanhe("1", uplo, n, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    // Check the imaginary parts of the diagonal elements and return with
    // an error code if any are nonzero.
    for (INT j = 0; j < n; j++) {
        if (cimagf(AFAC[j + j * ldafac]) != ZERO) {
            *resid = ONE / eps;
            return;
        }
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        // Compute the product U'*U, overwriting U.
        // Process columns from right to left.
        for (INT k = n - 1; k >= 0; k--) {
            // Compute the (k,k) element of the result.
            // Fortran: TR = DBLE( ZDOTC( K, AFAC(1,K), 1, AFAC(1,K), 1 ) )
            c64 tc;
            cblas_cdotc_sub(k + 1, &AFAC[k * ldafac], 1,
                            &AFAC[k * ldafac], 1, &tc);
            f32 tr = crealf(tc);
            AFAC[k + k * ldafac] = CMPLXF(tr, 0.0f);

            // Compute the rest of column k.
            // Fortran: ZTRMV('Upper','Conjugate','Non-unit', K-1, AFAC, LDAFAC, AFAC(1,K), 1)
            if (k > 0) {
                cblas_ctrmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    } else {
        // Compute the product L*L', overwriting L.
        // Process columns from right to left.
        for (INT k = n - 1; k >= 0; k--) {
            // Add a multiple of column k of the factor L to each of
            // columns k+1 through n-1.
            // Fortran: ZHER('Lower', N-K, ONE, AFAC(K+1,K), 1, AFAC(K+1,K+1), LDAFAC)
            if (k + 1 < n) {
                cblas_cher(CblasColMajor, CblasLower,
                           n - k - 1, ONE,
                           &AFAC[(k + 1) + k * ldafac], 1,
                           &AFAC[(k + 1) + (k + 1) * ldafac], ldafac);
            }

            // Scale column k by the diagonal element.
            // Fortran: TC = AFAC(K,K); ZSCAL(N-K+1, TC, AFAC(K,K), 1)
            c64 tc = AFAC[k + k * ldafac];
            cblas_cscal(n - k, &tc, &AFAC[k + k * ldafac], 1);
        }
    }

    // Compute the difference L*L' - A (or U'*U - A).
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < j; i++) {
                AFAC[i + j * ldafac] -= A[i + j * lda];
            }
            AFAC[j + j * ldafac] -= CMPLXF(crealf(A[j + j * lda]), 0.0f);
        }
    } else {
        for (INT j = 0; j < n; j++) {
            AFAC[j + j * ldafac] -= CMPLXF(crealf(A[j + j * lda]), 0.0f);
            for (INT i = j + 1; i < n; i++) {
                AFAC[i + j * ldafac] -= A[i + j * lda];
            }
        }
    }

    // Compute norm(L*L' - A) / (N * norm(A) * EPS)
    *resid = clanhe("1", uplo, n, AFAC, ldafac, rwork);
    *resid = ((*resid / (f32)n) / anorm) / eps;
}
