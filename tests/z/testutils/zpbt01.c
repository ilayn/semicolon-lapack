/**
 * @file zpbt01.c
 * @brief ZPBT01 reconstructs a Hermitian positive definite band matrix from
 *        its L*L' or U'*U factorization.
 *
 * Port of LAPACK TESTING/LIN/zpbt01.f
 */

#include <math.h>
#include "semicolon_lapack_complex_double.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZPBT01 reconstructs a Hermitian positive definite band matrix A from
 * its L*L' or U'*U factorization and computes the residual
 *    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
 *    norm( U'*U - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon, L' is the conjugate transpose of
 * L, and U' is the conjugate transpose of U.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the Hermitian matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     kd      The number of super-diagonals of the matrix A if
 *                        uplo = 'U', or the number of sub-diagonals if
 *                        uplo = 'L'. kd >= 0.
 * @param[in]     A       The original Hermitian band matrix A in band storage.
 *                        Dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= kd+1.
 * @param[in,out] AFAC    On entry, the factored form of A from ZPBTRF.
 *                        On exit, overwritten by L*L' or U'*U.
 *                        Dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= kd+1.
 * @param[out]    rwork   Workspace array, dimension (n).
 * @param[out]    resid   If uplo = 'L', norm(L*L' - A) / ( N * norm(A) * EPS )
 *                        If uplo = 'U', norm(U'*U - A) / ( N * norm(A) * EPS )
 */
void zpbt01(const char* uplo, const INT n, const INT kd,
            const c128* A, const INT lda,
            c128* AFAC, const INT ldafac,
            f64* rwork, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *resid = ZERO;
    if (n <= 0) {
        return;
    }

    f64 eps = dlamch("Epsilon");
    f64 anorm = zlanhb("1", uplo, n, kd, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT j = 0; j < n; j++) {
            if (cimag(AFAC[kd + j * ldafac]) != ZERO) {
                *resid = ONE / eps;
                return;
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            if (cimag(AFAC[0 + j * ldafac]) != ZERO) {
                *resid = ONE / eps;
                return;
            }
        }
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT k = n - 1; k >= 0; k--) {
            INT kc = (kd - k > 0) ? kd - k : 0;
            INT klen = kd - kc;

            c128 zdotc;
            cblas_zdotc_sub(klen + 1, &AFAC[kc + k * ldafac], 1,
                            &AFAC[kc + k * ldafac], 1, &zdotc);
            f64 akk = creal(zdotc);
            AFAC[kd + k * ldafac] = akk;

            if (klen > 0) {
                cblas_ztrmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
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
                cblas_zher(CblasColMajor, CblasLower, klen, ONE,
                           &AFAC[1 + k * ldafac], 1,
                           &AFAC[0 + (k + 1) * ldafac], ldafac - 1);
            }

            f64 akk = creal(AFAC[0 + k * ldafac]);
            cblas_zdscal(klen + 1, akk, &AFAC[0 + k * ldafac], 1);
        }

        for (INT j = 0; j < n; j++) {
            INT ml = (kd + 1 < n - j) ? kd + 1 : n - j;
            for (INT i = 0; i < ml; i++) {
                AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
            }
        }
    }

    *resid = zlanhb("I", uplo, n, kd, AFAC, ldafac, rwork);

    *resid = ((*resid / (f64)n) / anorm) / eps;
}
