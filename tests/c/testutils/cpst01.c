/**
 * @file cpst01.c
 * @brief CPST01 reconstructs a Hermitian positive semidefinite matrix from
 *        its L or U factors and the permutation matrix P and computes the
 *        residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CPST01 reconstructs a Hermitian positive semidefinite matrix A
 * from its L or U factors and the permutation matrix P and computes
 * the residual
 *    norm( P*L*L'*P' - A ) / ( N * norm(A) * EPS ) or
 *    norm( P*U'*U*P' - A ) / ( N * norm(A) * EPS ),
 * where EPS is the machine epsilon, L' is the conjugate transpose of L,
 * and U' is the conjugate transpose of U.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the Hermitian matrix A is stored:
 *                        = 'U':  Upper triangular
 *                        = 'L':  Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       Complex array, dimension (lda, n).
 *                        The original Hermitian matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     AFAC    Complex array, dimension (ldafac, n).
 *                        The factor L or U from the L*L' or U'*U factorization
 *                        of A.
 * @param[in]     ldafac  The leading dimension of the array AFAC.
 *                        ldafac >= max(1,n).
 * @param[out]    PERM    Complex array, dimension (ldperm, n).
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
void cpst01(
    const char* uplo,
    const INT n,
    const c64* const restrict A,
    const INT lda,
    c64* const restrict AFAC,
    const INT ldafac,
    c64* const restrict PERM,
    const INT ldperm,
    const INT* const restrict piv,
    f32* const restrict rwork,
    f32* resid,
    const INT rank)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    f32 eps = slamch("E");
    f32 anorm = clanhe("1", uplo, n, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Check the imaginary parts of the diagonal elements and return with
       an error code if any are nonzero. */
    for (INT j = 0; j < n; j++) {
        if (cimagf(AFAC[j + j * ldafac]) != ZERO) {
            *resid = ONE / eps;
            return;
        }
    }

    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        if (rank < n) {
            for (INT j = rank; j < n; j++) {
                for (INT i = rank; i <= j; i++) {
                    AFAC[i + j * ldafac] = CZERO;
                }
            }
        }

        for (INT k = n - 1; k >= 0; k--) {
            c64 tc;
            cblas_cdotc_sub(k + 1, &AFAC[k * ldafac], 1,
                                    &AFAC[k * ldafac], 1, &tc);
            f32 tr = crealf(tc);
            AFAC[k + k * ldafac] = CMPLXF(tr, 0.0f);

            if (k > 0) {
                cblas_ctrmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    } else {
        if (rank < n) {
            for (INT j = rank; j < n; j++) {
                for (INT i = j; i < n; i++) {
                    AFAC[i + j * ldafac] = CZERO;
                }
            }
        }

        for (INT k = n - 1; k >= 0; k--) {
            if (k + 1 < n) {
                cblas_cher(CblasColMajor, CblasLower, n - k - 1, ONE,
                           &AFAC[(k + 1) + k * ldafac], 1,
                           &AFAC[(k + 1) + (k + 1) * ldafac], ldafac);
            }

            c64 tc = AFAC[k + k * ldafac];
            cblas_cscal(n - k, &tc, &AFAC[k + k * ldafac], 1);
        }
    }

    if (upper) {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (piv[i] <= piv[j]) {
                    if (i <= j) {
                        PERM[piv[i] + piv[j] * ldperm] = AFAC[i + j * ldafac];
                    } else {
                        PERM[piv[i] + piv[j] * ldperm] = conjf(AFAC[j + i * ldafac]);
                    }
                }
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (piv[i] >= piv[j]) {
                    if (i >= j) {
                        PERM[piv[i] + piv[j] * ldperm] = AFAC[i + j * ldafac];
                    } else {
                        PERM[piv[i] + piv[j] * ldperm] = conjf(AFAC[j + i * ldafac]);
                    }
                }
            }
        }
    }

    if (upper) {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < j; i++) {
                PERM[i + j * ldperm] = PERM[i + j * ldperm] - A[i + j * lda];
            }
            PERM[j + j * ldperm] = PERM[j + j * ldperm] - CMPLXF(crealf(A[j + j * lda]), 0.0f);
        }
    } else {
        for (INT j = 0; j < n; j++) {
            PERM[j + j * ldperm] = PERM[j + j * ldperm] - CMPLXF(crealf(A[j + j * lda]), 0.0f);
            for (INT i = j + 1; i < n; i++) {
                PERM[i + j * ldperm] = PERM[i + j * ldperm] - A[i + j * lda];
            }
        }
    }

    *resid = clanhe("1", uplo, n, PERM, ldperm, rwork);

    *resid = ((*resid / (f32)n) / anorm) / eps;
}
