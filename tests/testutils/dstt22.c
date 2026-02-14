/**
 * @file dstt22.c
 * @brief DSTT22 checks a set of M eigenvalues and eigenvectors
 *        of a symmetric tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/EIG/dstt22.f to C.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);

/**
 * DSTT22 checks a set of M eigenvalues and eigenvectors,
 *
 *    A U = U S
 *
 * where A is symmetric tridiagonal, the columns of U are orthogonal,
 * and S is diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
 * Two tests are performed:
 *
 *    RESULT(1) = | U' A U - S | / ( |A| m ulp )
 *
 *    RESULT(2) = | I - U'U | / ( m ulp )
 *
 * @param[in]     n       The size of the matrix. If zero, does nothing.
 * @param[in]     m       The number of eigenpairs to check. If zero, does nothing.
 * @param[in]     kband   The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     AD      Diagonal of A, dimension (n).
 * @param[in]     AE      Off-diagonal of A, dimension (n).
 * @param[in]     SD      Diagonal of S, dimension (n).
 * @param[in]     SE      Off-diagonal of S, dimension (n). Not referenced if kband=0.
 * @param[in]     U       The n by m orthogonal matrix U, dimension (ldu, m).
 * @param[in]     ldu     Leading dimension of U. ldu >= n.
 * @param[out]    work    Workspace array, dimension (ldwork, m+1).
 * @param[in]     ldwork  Leading dimension of work. ldwork >= max(1, m).
 * @param[out]    result  Array of dimension (2). The two test ratios.
 */
void dstt22(const int n, const int m, const int kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const f64* const restrict U, const int ldu,
            f64* const restrict work, const int ldwork,
            f64* restrict result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i, j, k;
    f64 anorm, aukj, ulp, unfl, wnorm;

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0 || m <= 0)
        return;

    unfl = dlamch("S");
    ulp = dlamch("E");

    /* Do Test 1 */

    /* Compute the 1-norm of A. */
    if (n > 1) {
        anorm = fabs(AD[0]) + fabs(AE[0]);
        for (j = 1; j < n - 1; j++) {
            anorm = fmax(anorm, fabs(AD[j]) + fabs(AE[j]) + fabs(AE[j - 1]));
        }
        anorm = fmax(anorm, fabs(AD[n - 1]) + fabs(AE[n - 2]));
    } else {
        anorm = fabs(AD[0]);
    }
    anorm = fmax(anorm, unfl);

    /* Norm of U'AU - S */
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            work[i + j * ldwork] = ZERO;
            for (k = 0; k < n; k++) {
                aukj = AD[k] * U[k + j * ldu];
                if (k != n - 1)
                    aukj = aukj + AE[k] * U[k + 1 + j * ldu];
                if (k != 0)
                    aukj = aukj + AE[k - 1] * U[k - 1 + j * ldu];
                work[i + j * ldwork] = work[i + j * ldwork] + U[k + i * ldu] * aukj;
            }
        }
        work[i + i * ldwork] = work[i + i * ldwork] - SD[i];
        if (kband == 1) {
            if (i != 0)
                work[i + (i - 1) * ldwork] = work[i + (i - 1) * ldwork] - SE[i - 1];
            if (i != n - 1)
                work[i + (i + 1) * ldwork] = work[i + (i + 1) * ldwork] - SE[i];
        }
    }

    wnorm = dlansy("1", "L", m, work, m, &work[m * ldwork]);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / ((f64)m * ulp);
    } else {
        if (anorm < ONE) {
            result[0] = (fmin(wnorm, (f64)m * anorm) / anorm) / ((f64)m * ulp);
        } else {
            result[0] = fmin(wnorm / anorm, (f64)m) / ((f64)m * ulp);
        }
    }

    /* Do Test 2 */

    /* Compute  U'U - I */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, m, n, ONE, U, ldu, U, ldu, ZERO, work, m);

    for (j = 0; j < m; j++) {
        work[j + j * m] = work[j + j * m] - ONE;
    }

    result[1] = fmin((f64)m,
                     dlange("1", m, m, work, m, &work[m * ldwork])) / ((f64)m * ulp);
}
