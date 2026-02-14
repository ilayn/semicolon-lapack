/**
 * @file dget01.c
 * @brief DGET01 reconstructs a matrix A from its L*U factorization and
 *        computes the residual.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64 * const restrict A, const int lda,
                     f64 * const restrict work);
extern void dlaswp(const int n, f64 * const restrict A, const int lda,
                   const int k1, const int k2,
                   const int * const restrict ipiv, const int incx);

/**
 * DGET01 reconstructs a matrix A from its L*U factorization and
 * computes the residual
 *    norm(L*U - A) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original m x n matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,m).
 * @param[in,out] AFAC    Double precision array, dimension (ldafac, n).
 *                        The factored form of the matrix A. AFAC contains the
 *                        factors L and U from the L*U factorization as computed
 *                        by dgetrf. Overwritten with the reconstructed matrix,
 *                        and then with the difference L*U - A.
 * @param[in]     ldafac  The leading dimension of the array AFAC.
 *                        ldafac >= max(1,m).
 * @param[in]     ipiv    Integer array, dimension (min(m,n)).
 *                        The pivot indices from dgetrf. 0-based indexing.
 * @param[out]    rwork   Double precision array, dimension (m).
 * @param[out]    resid   norm(L*U - A) / ( N * norm(A) * EPS )
 */
void dget01(
    const int m,
    const int n,
    const f64 * const restrict A,
    const int lda,
    f64 * const restrict AFAC,
    const int ldafac,
    const int * const restrict ipiv,
    f64 * const restrict rwork,
    f64 *resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i, j, k;
    f64 anorm, eps, t;
    int minmn;

    // Quick exit if m = 0 or n = 0
    if (m <= 0 || n <= 0) {
        *resid = ZERO;
        return;
    }

    // Determine EPS and the norm of A
    eps = dlamch("E");
    anorm = dlange("1", m, n, A, lda, rwork);

    // Compute the product L*U and overwrite AFAC with the result.
    // A column at a time of the product is obtained, starting with column N-1.

    minmn = (m < n) ? m : n;

    for (k = n - 1; k >= 0; k--) {
        if (k >= m) {
            // DTRMV: AFAC(0:m-1, k) = L * AFAC(0:m-1, k)
            // L is m x m unit lower triangular
            cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        m, AFAC, ldafac, &AFAC[k * ldafac], 1);
        } else {
            // Compute elements (k+1:m-1, k)
            t = AFAC[k + k * ldafac];
            if (k + 1 < m) {
                // Scale: AFAC(k+1:m-1, k) *= t
                cblas_dscal(m - k - 1, t, &AFAC[k + 1 + k * ldafac], 1);
                // DGEMV: AFAC(k+1:m-1, k) += L(k+1:m-1, 0:k-1) * U(0:k-1, k)
                if (k > 0) {
                    cblas_dgemv(CblasColMajor, CblasNoTrans, m - k - 1, k, ONE,
                                &AFAC[k + 1], ldafac, &AFAC[k * ldafac], 1, ONE,
                                &AFAC[k + 1 + k * ldafac], 1);
                }
            }

            // Compute the (k,k) element
            // AFAC(k,k) = t + dot(L(k, 0:k-1), U(0:k-1, k))
            if (k > 0) {
                AFAC[k + k * ldafac] = t + cblas_ddot(k, &AFAC[k], ldafac,
                                                       &AFAC[k * ldafac], 1);
            }

            // Compute elements (0:k-1, k)
            // DTRMV: AFAC(0:k-1, k) = L(0:k-1, 0:k-1) * AFAC(0:k-1, k)
            if (k > 0) {
                cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    }

    // Apply inverse row permutation: DLASWP with incx = -1
    dlaswp(n, AFAC, ldafac, 0, minmn - 1, ipiv, -1);

    // Compute the difference L*U - A and store in AFAC
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
        }
    }

    // Compute norm(L*U - A) / ( N * norm(A) * EPS )
    *resid = dlange("1", m, n, AFAC, ldafac, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
