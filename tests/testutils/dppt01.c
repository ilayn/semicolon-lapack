/**
 * @file dppt01.c
 * @brief DPPT01 reconstructs a symmetric positive definite packed matrix A from its factorization.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

extern f64 dlamch(const char* cmach);
extern f64 dlansp(const char* norm, const char* uplo, const int n,
                     const f64* const restrict AP,
                     f64* const restrict work);

/**
 * DPPT01 reconstructs a symmetric positive definite packed matrix A
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
void dppt01(const char* uplo, const int n,
            const f64* const restrict A,
            f64* const restrict AFAC,
            f64* const restrict rwork,
            f64* resid)
{
    int i, k, kc, npp;
    f64 anorm, eps, t;

    if (n <= 0) {
        *resid = 0.0;
        return;
    }

    eps = dlamch("E");
    anorm = dlansp("1", uplo, n, A, rwork);
    if (anorm <= 0.0) {
        *resid = 1.0 / eps;
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        kc = (n * (n - 1)) / 2;
        for (k = n; k >= 1; k--) {

            t = cblas_ddot(k, &AFAC[kc], 1, &AFAC[kc], 1);
            AFAC[kc + k - 1] = t;

            if (k > 1) {
                cblas_dtpmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            k - 1, AFAC, &AFAC[kc], 1);
                kc = kc - (k - 1);
            }
        }

    } else {
        kc = (n * (n + 1)) / 2 - 1;
        for (k = n; k >= 1; k--) {

            if (k < n) {
                cblas_dspr(CblasColMajor, CblasLower, n - k, 1.0,
                           &AFAC[kc + 1], 1, &AFAC[kc + n - k + 1]);
            }

            t = AFAC[kc];
            cblas_dscal(n - k + 1, t, &AFAC[kc], 1);

            kc = kc - (n - k + 2);
        }
    }

    npp = n * (n + 1) / 2;
    for (i = 0; i < npp; i++) {
        AFAC[i] = AFAC[i] - A[i];
    }

    *resid = dlansp("1", uplo, n, AFAC, rwork);

    *resid = ((*resid / (f64)n) / anorm) / eps;
}
