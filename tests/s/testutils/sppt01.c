/**
 * @file sppt01.c
 * @brief SPPT01 reconstructs a symmetric positive definite packed matrix A from its factorization.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

extern f32 slamch(const char* cmach);
extern f32 slansp(const char* norm, const char* uplo, const int n,
                     const f32* const restrict AP,
                     f32* const restrict work);

/**
 * SPPT01 reconstructs a symmetric positive definite packed matrix A
 * from its L*L' or U'*U factorization and computes the residual
 *    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
 *    norm( U'*U - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The number of rows and columns of the matrix A. n >= 0.
 *
 * @param[in] A
 *          The original symmetric matrix A, stored as a packed
 *          triangular matrix. Dimension (n*(n+1)/2).
 *
 * @param[in,out] AFAC
 *          On entry, the factor L or U from the L*L' or U'*U
 *          factorization of A, stored as a packed triangular matrix.
 *          Overwritten with the reconstructed matrix, and then with the
 *          difference L*L' - A (or U'*U - A). Dimension (n*(n+1)/2).
 *
 * @param[out] rwork
 *          Workspace of dimension (n).
 *
 * @param[out] resid
 *          If uplo = 'L', norm(L*L' - A) / ( N * norm(A) * EPS )
 *          If uplo = 'U', norm(U'*U - A) / ( N * norm(A) * EPS )
 */
void sppt01(const char* uplo, const int n,
            const f32* const restrict A,
            f32* const restrict AFAC,
            f32* const restrict rwork,
            f32* resid)
{
    int i, k, kc, npp;
    f32 anorm, eps, t;

    if (n <= 0) {
        *resid = 0.0f;
        return;
    }

    eps = slamch("E");
    anorm = slansp("1", uplo, n, A, rwork);
    if (anorm <= 0.0f) {
        *resid = 1.0f / eps;
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        kc = (n * (n - 1)) / 2;
        for (k = n; k >= 1; k--) {

            t = cblas_sdot(k, &AFAC[kc], 1, &AFAC[kc], 1);
            AFAC[kc + k - 1] = t;

            if (k > 1) {
                cblas_stpmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            k - 1, AFAC, &AFAC[kc], 1);
                kc = kc - (k - 1);
            }
        }

    } else {
        kc = (n * (n + 1)) / 2 - 1;
        for (k = n; k >= 1; k--) {

            if (k < n) {
                cblas_sspr(CblasColMajor, CblasLower, n - k, 1.0f,
                           &AFAC[kc + 1], 1, &AFAC[kc + n - k + 1]);
            }

            t = AFAC[kc];
            cblas_sscal(n - k + 1, t, &AFAC[kc], 1);

            kc = kc - (n - k + 2);
        }
    }

    npp = n * (n + 1) / 2;
    for (i = 0; i < npp; i++) {
        AFAC[i] = AFAC[i] - A[i];
    }

    *resid = slansp("1", uplo, n, AFAC, rwork);

    *resid = ((*resid / (f32)n) / anorm) / eps;
}
