/**
 * @file dtpt01.c
 * @brief DTPT01 computes the residual for a triangular matrix A times its inverse
 *        when A is stored in packed format.
 *
 * Port of LAPACK TESTING/LIN/dtpt01.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlantp(const char* norm, const char* uplo, const char* diag,
                     const int n, const f64* AP, f64* work);

/**
 * DTPT01 computes the residual for a triangular matrix A times its inverse
 * when A is stored in packed format:
 *    RESID = norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     AP      Array (n*(n+1)/2). The triangular matrix A in packed storage.
 * @param[in,out] AINVP   Array (n*(n+1)/2). On entry, the inverse of A in packed storage.
 *                        On exit, the contents are destroyed.
 * @param[out]    rcond   The reciprocal condition number = 1/(norm(A) * norm(AINV)).
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS).
 */
void dtpt01(const char* uplo, const char* diag, const int n,
            const f64* AP, f64* AINVP,
            f64* rcond, f64* work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    int j, jc;
    f64 ainvnm, anorm, eps;
    int unitd;

    /* Quick exit if N = 0 */
    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    /* Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0 */
    eps = dlamch("E");
    anorm = dlantp("1", uplo, diag, n, AP, work);
    ainvnm = dlantp("1", uplo, diag, n, AINVP, work);

    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    /* Compute A * AINV, overwriting AINV */
    unitd = (diag[0] == 'U' || diag[0] == 'u');

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        jc = 0;
        for (j = 0; j < n; j++) {
            if (unitd) {
                AINVP[jc + j] = ONE;
            }

            /* Form the j-th column of A*AINV using dtpmv */
            cblas_dtpmv(CblasColMajor, CblasUpper, CblasNoTrans,
                       unitd ? CblasUnit : CblasNonUnit,
                       j + 1, AP, &AINVP[jc], 1);

            /* Subtract 1 from the diagonal */
            AINVP[jc + j] -= ONE;
            jc += j + 1;
        }
    } else {
        jc = 0;
        for (j = 0; j < n; j++) {
            if (unitd) {
                AINVP[jc] = ONE;
            }

            /* Form the j-th column of A*AINV using dtpmv */
            cblas_dtpmv(CblasColMajor, CblasLower, CblasNoTrans,
                       unitd ? CblasUnit : CblasNonUnit,
                       n - j, &AP[jc], &AINVP[jc], 1);

            /* Subtract 1 from the diagonal */
            AINVP[jc] -= ONE;
            jc += n - j;
        }
    }

    /* Compute norm(A*AINV - I) / (N * norm(A) * norm(AINV) * EPS) */
    *resid = dlantp("1", uplo, "N", n, AINVP, work);
    *resid = ((*resid) * (*rcond) / (f64)n) / eps;
}
