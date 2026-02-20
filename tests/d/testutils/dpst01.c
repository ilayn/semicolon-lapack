/**
 * @file dpst01.c
 * @brief DPST01 reconstructs a symmetric positive semidefinite matrix from
 *        its L or U factors and the permutation matrix P and computes the
 *        residual.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

extern f64 dlamch(const char* cmach);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* A, const int lda, f64* work);

/**
 * DPST01 reconstructs a symmetric positive semidefinite matrix A
 * from its L or U factors and the permutation matrix P and computes
 * the residual
 *    norm( P*L*L'*P' - A ) / ( N * norm(A) * EPS ) or
 *    norm( P*U'*U*P' - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored:
 *                        = 'U':  Upper triangular
 *                        = 'L':  Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     AFAC    Double precision array, dimension (ldafac, n).
 *                        The factor L or U from the L*L' or U'*U factorization
 *                        of A.
 * @param[in]     ldafac  The leading dimension of the array AFAC.
 *                        ldafac >= max(1,n).
 * @param[out]    PERM    Double precision array, dimension (ldperm, n).
 *                        Overwritten with the reconstructed matrix, and then
 *                        with the difference P*L*L'*P' - A (or P*U'*U*P' - A).
 * @param[in]     ldperm  The leading dimension of the array PERM.
 *                        ldperm >= max(1,n).
 * @param[in]     piv     Integer array, dimension (n).
 *                        PIV is such that the nonzero entries are
 *                        P(PIV(k), k) = 1. 0-based indexing.
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   If UPLO = 'L', norm(L*L' - A) / (N * norm(A) * EPS)
 *                        If UPLO = 'U', norm(U'*U - A) / (N * norm(A) * EPS)
 * @param[in]     rank    Number of nonzero singular values of A.
 */
void dpst01(
    const char* uplo,
    const int n,
    const f64* const restrict A,
    const int lda,
    f64* const restrict AFAC,
    const int ldafac,
    f64* const restrict PERM,
    const int ldperm,
    const int* const restrict piv,
    f64* const restrict rwork,
    f64* resid,
    const int rank)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    f64 eps = dlamch("E");
    f64 anorm = dlansy("1", uplo, n, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    int upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        if (rank < n) {
            for (int j = rank; j < n; j++) {
                for (int i = rank; i <= j; i++) {
                    AFAC[i + j * ldafac] = ZERO;
                }
            }
        }

        for (int k = n - 1; k >= 0; k--) {
            f64 t = cblas_ddot(k + 1, &AFAC[k * ldafac], 1,
                                  &AFAC[k * ldafac], 1);
            AFAC[k + k * ldafac] = t;

            if (k > 0) {
                cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    } else {
        if (rank < n) {
            for (int j = rank; j < n; j++) {
                for (int i = j; i < n; i++) {
                    AFAC[i + j * ldafac] = ZERO;
                }
            }
        }

        for (int k = n - 1; k >= 0; k--) {
            if (k + 1 < n) {
                cblas_dsyr(CblasColMajor, CblasLower, n - k - 1, ONE,
                           &AFAC[(k + 1) + k * ldafac], 1,
                           &AFAC[(k + 1) + (k + 1) * ldafac], ldafac);
            }

            f64 t = AFAC[k + k * ldafac];
            cblas_dscal(n - k, t, &AFAC[k + k * ldafac], 1);
        }
    }

    if (upper) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                if (piv[i] <= piv[j]) {
                    if (i <= j) {
                        PERM[piv[i] + piv[j] * ldperm] = AFAC[i + j * ldafac];
                    } else {
                        PERM[piv[i] + piv[j] * ldperm] = AFAC[j + i * ldafac];
                    }
                }
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                if (piv[i] >= piv[j]) {
                    if (i >= j) {
                        PERM[piv[i] + piv[j] * ldperm] = AFAC[i + j * ldafac];
                    } else {
                        PERM[piv[i] + piv[j] * ldperm] = AFAC[j + i * ldafac];
                    }
                }
            }
        }
    }

    if (upper) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                PERM[i + j * ldperm] = PERM[i + j * ldperm] - A[i + j * lda];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                PERM[i + j * ldperm] = PERM[i + j * ldperm] - A[i + j * lda];
            }
        }
    }

    *resid = dlansy("1", uplo, n, PERM, ldperm, rwork);

    *resid = ((*resid / (f64)n) / anorm) / eps;
}
