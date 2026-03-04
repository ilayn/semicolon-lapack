/**
 * @file zppt01.c
 * @brief ZPPT01 reconstructs a Hermitian positive definite packed matrix A from its factorization.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZPPT01 reconstructs a Hermitian positive definite packed matrix A
 * from its L*L' or U'*U factorization and computes the residual
 *    norm( L*L' - A ) / ( N * norm(A) * EPS ) or
 *    norm( U'*U - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon, L' is the conjugate transpose of
 * L, and U' is the conjugate transpose of U.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The number of rows and columns of the matrix A. n >= 0.
 *
 * @param[in] A
 *          The original Hermitian matrix A, stored as a packed
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
void zppt01(const char* uplo, const INT n,
            const c128* const restrict A,
            c128* const restrict AFAC,
            f64* const restrict rwork,
            f64* resid)
{
    INT i, k, kc;
    f64 anorm, eps, tr;
    c128 tc;

    if (n <= 0) {
        *resid = 0.0;
        return;
    }

    eps = dlamch("E");
    anorm = zlanhp("1", uplo, n, A, rwork);
    if (anorm <= 0.0) {
        *resid = 1.0 / eps;
        return;
    }

    /* Check the imaginary parts of the diagonal elements and return with
     * an error code if any are nonzero. */
    kc = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (k = 0; k < n; k++) {
            if (cimag(AFAC[kc]) != 0.0) {
                *resid = 1.0 / eps;
                return;
            }
            kc = kc + k + 2;
        }
    } else {
        for (k = 0; k < n; k++) {
            if (cimag(AFAC[kc]) != 0.0) {
                *resid = 1.0 / eps;
                return;
            }
            kc = kc + n - k;
        }
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Compute the product U'*U, overwriting U. */
        kc = (n * (n - 1)) / 2;
        for (k = n; k >= 1; k--) {

            /* Compute the (K,K) element of the result. */
            c128 dotc;
            cblas_zdotc_sub(k, &AFAC[kc], 1, &AFAC[kc], 1, &dotc);
            tr = creal(dotc);
            AFAC[kc + k - 1] = CMPLX(tr, 0.0);

            /* Compute the rest of column K. */
            if (k > 1) {
                cblas_ztpmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                            k - 1, AFAC, &AFAC[kc], 1);
                kc = kc - (k - 1);
            }
        }

        /* Compute the difference U'*U - A */
        kc = 0;
        for (k = 0; k < n; k++) {
            for (i = 0; i < k; i++) {
                AFAC[kc + i] = AFAC[kc + i] - A[kc + i];
            }
            AFAC[kc + k] = AFAC[kc + k] - creal(A[kc + k]);
            kc = kc + k + 1;
        }

    } else {
        /* Compute the product L*L', overwriting L. */
        kc = (n * (n + 1)) / 2 - 1;
        for (k = n; k >= 1; k--) {

            if (k < n) {
                cblas_zhpr(CblasColMajor, CblasLower, n - k, 1.0,
                           &AFAC[kc + 1], 1, &AFAC[kc + n - k + 1]);
            }

            /* Scale column K by the diagonal element. */
            tc = AFAC[kc];
            cblas_zscal(n - k + 1, &tc, &AFAC[kc], 1);

            kc = kc - (n - k + 2);
        }

        /* Compute the difference L*L' - A */
        kc = 0;
        for (k = 0; k < n; k++) {
            AFAC[kc] = AFAC[kc] - creal(A[kc]);
            for (i = k + 1; i < n; i++) {
                AFAC[kc + i - k] = AFAC[kc + i - k] - A[kc + i - k];
            }
            kc = kc + n - k;
        }
    }

    *resid = zlanhp("1", uplo, n, AFAC, rwork);

    *resid = ((*resid / (f64)n) / anorm) / eps;
}
