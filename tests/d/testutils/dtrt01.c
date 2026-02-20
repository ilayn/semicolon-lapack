/**
 * @file dtrt01.c
 * @brief DTRT01 computes the residual for a triangular matrix A times its inverse.
 *
 * Port of LAPACK TESTING/LIN/dtrt01.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlantr(const char* norm, const char* uplo, const char* diag,
                     const int m, const int n, const f64* A, const int lda,
                     f64* work);

/**
 * DTRT01 computes the residual for a triangular matrix A times its inverse:
 *    RESID = norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     A       Array (lda, n). The triangular matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1, n).
 * @param[in,out] AINV    Array (ldainv, n). On entry, the inverse of A.
 *                        On exit, the contents are destroyed.
 * @param[in]     ldainv  The leading dimension of the array AINV.
 * @param[out]    rcond   The reciprocal condition number = 1/(norm(A) * norm(AINV)).
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS).
 */
void dtrt01(const char* uplo, const char* diag, const int n,
            const f64* A, const int lda,
            f64* AINV, const int ldainv,
            f64* rcond, f64* work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    int j;
    f64 ainvnm, anorm, eps;

    /* Quick exit if N = 0 */
    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    /* Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0 */
    eps = dlamch("E");
    anorm = dlantr("1", uplo, diag, n, n, A, lda, work);
    ainvnm = dlantr("1", uplo, diag, n, n, AINV, ldainv, work);

    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    /* Set the diagonal of AINV to 1 if AINV has unit diagonal */
    if (diag[0] == 'U' || diag[0] == 'u') {
        for (j = 0; j < n; j++) {
            AINV[j * ldainv + j] = ONE;
        }
    }

    /* Compute A * AINV, overwriting AINV */
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Upper triangular */
        for (j = 0; j < n; j++) {
            /* A is (j+1) x (j+1) upper triangular for column j */
            cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans,
                       (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                       j + 1, A, lda, &AINV[j * ldainv], 1);
        }
    } else {
        /* Lower triangular */
        for (j = 0; j < n; j++) {
            /* A(j:n-1, j:n-1) is lower triangular for column j */
            cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans,
                       (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit,
                       n - j, &A[j * lda + j], lda, &AINV[j * ldainv + j], 1);
        }
    }

    /* Subtract 1 from each diagonal element to form A*AINV - I */
    for (j = 0; j < n; j++) {
        AINV[j * ldainv + j] -= ONE;
    }

    /* Compute norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS) */
    *resid = dlantr("1", uplo, "N", n, n, AINV, ldainv, work);
    *resid = ((*resid) * (*rcond) / (f64)n) / eps;
}
