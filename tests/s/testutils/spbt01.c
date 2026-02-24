/**
 * @file spbt01.c
 * @brief SPBT01 reconstructs a symmetric positive definite band matrix from
 *        its L*L' or U'*U factorization.
 *
 * Port of LAPACK TESTING/LIN/spbt01.f
 */

#include <math.h>
#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SPBT01 reconstructs a symmetric positive definite band matrix A from
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
 * @param[in]     kd      The number of super-diagonals of the matrix A if
 *                        uplo = 'U', or the number of sub-diagonals if
 *                        uplo = 'L'. kd >= 0.
 * @param[in]     A       The original symmetric band matrix A in band storage.
 *                        Dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= kd+1.
 * @param[in,out] AFAC    On entry, the factored form of A from SPBTRF.
 *                        On exit, overwritten by L*L' or U'*U.
 *                        Dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= kd+1.
 * @param[out]    rwork   Workspace array, dimension (n).
 * @param[out]    resid   If uplo = 'L', norm(L*L' - A) / ( N * norm(A) * EPS )
 *                        If uplo = 'U', norm(U'*U - A) / ( N * norm(A) * EPS )
 */
void spbt01(const char* uplo, const INT n, const INT kd,
            const f32* A, const INT lda,
            f32* AFAC, const INT ldafac,
            f32* rwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    *resid = ZERO;
    if (n <= 0) {
        return;
    }

    f32 eps = slamch("Epsilon");
    f32 anorm = slansb("1", uplo, n, kd, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT k = n - 1; k >= 0; k--) {
            INT kc = (kd - k > 0) ? kd - k : 0;
            INT klen = kd - kc;

            f32 t = cblas_sdot(klen + 1, &AFAC[kc + k * ldafac], 1,
                                  &AFAC[kc + k * ldafac], 1);
            AFAC[kd + k * ldafac] = t;

            if (klen > 0) {
                cblas_strmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            klen, &AFAC[kd + (k - klen) * ldafac], ldafac - 1,
                            &AFAC[kc + k * ldafac], 1);
            }
        }

        for (INT j = 0; j < n; j++) {
            INT mu = (kd - j > 0) ? kd - j : 0;
            for (INT i = mu; i <= kd; i++) {
                AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
            }
        }
    } else {
        for (INT k = n - 1; k >= 0; k--) {
            INT klen = (kd < n - k - 1) ? kd : n - k - 1;

            if (klen > 0) {
                cblas_ssyr(CblasColMajor, CblasLower, klen, ONE,
                           &AFAC[1 + k * ldafac], 1,
                           &AFAC[0 + (k + 1) * ldafac], ldafac - 1);
            }

            f32 t = AFAC[0 + k * ldafac];
            cblas_sscal(klen + 1, t, &AFAC[0 + k * ldafac], 1);
        }

        for (INT j = 0; j < n; j++) {
            INT ml = (kd + 1 < n - j) ? kd + 1 : n - j;
            for (INT i = 0; i < ml; i++) {
                AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
            }
        }
    }

    *resid = slansb("I", uplo, n, kd, AFAC, ldafac, rwork);

    *resid = ((*resid / (f32)n) / anorm) / eps;
}
