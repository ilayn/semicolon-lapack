/**
 * @file dpbt01.c
 * @brief DPBT01 reconstructs a symmetric positive definite band matrix from
 *        its L*L' or U'*U factorization.
 *
 * Port of LAPACK TESTING/LIN/dpbt01.f
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

/**
 * DPBT01 reconstructs a symmetric positive definite band matrix A from
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
 * @param[in,out] AFAC    On entry, the factored form of A from DPBTRF.
 *                        On exit, overwritten by L*L' or U'*U.
 *                        Dimension (ldafac, n).
 * @param[in]     ldafac  The leading dimension of AFAC. ldafac >= kd+1.
 * @param[out]    rwork   Workspace array, dimension (n).
 * @param[out]    resid   If uplo = 'L', norm(L*L' - A) / ( N * norm(A) * EPS )
 *                        If uplo = 'U', norm(U'*U - A) / ( N * norm(A) * EPS )
 */
void dpbt01(const char* uplo, const int n, const int kd,
            const double* A, const int lda,
            double* AFAC, const int ldafac,
            double* rwork, double* resid)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    *resid = ZERO;
    if (n <= 0) {
        return;
    }

    double eps = dlamch("Epsilon");
    double anorm = dlansb("1", uplo, n, kd, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (int k = n - 1; k >= 0; k--) {
            int kc = (kd - k > 0) ? kd - k : 0;
            int klen = kd - kc;

            double t = cblas_ddot(klen + 1, &AFAC[kc + k * ldafac], 1,
                                  &AFAC[kc + k * ldafac], 1);
            AFAC[kd + k * ldafac] = t;

            if (klen > 0) {
                cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            klen, &AFAC[kd + (k - klen) * ldafac], ldafac - 1,
                            &AFAC[kc + k * ldafac], 1);
            }
        }

        for (int j = 0; j < n; j++) {
            int mu = (kd - j > 0) ? kd - j : 0;
            for (int i = mu; i <= kd; i++) {
                AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
            }
        }
    } else {
        for (int k = n - 1; k >= 0; k--) {
            int klen = (kd < n - k - 1) ? kd : n - k - 1;

            if (klen > 0) {
                cblas_dsyr(CblasColMajor, CblasLower, klen, ONE,
                           &AFAC[1 + k * ldafac], 1,
                           &AFAC[0 + (k + 1) * ldafac], ldafac - 1);
            }

            double t = AFAC[0 + k * ldafac];
            cblas_dscal(klen + 1, t, &AFAC[0 + k * ldafac], 1);
        }

        for (int j = 0; j < n; j++) {
            int ml = (kd + 1 < n - j) ? kd + 1 : n - j;
            for (int i = 0; i < ml; i++) {
                AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
            }
        }
    }

    *resid = dlansb("I", uplo, n, kd, AFAC, ldafac, rwork);

    *resid = ((*resid / (double)n) / anorm) / eps;
}
