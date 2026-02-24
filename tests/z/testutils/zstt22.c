/**
 * @file zstt22.c
 * @brief ZSTT22 checks a set of M eigenvalues and eigenvectors
 *        of a Hermitian tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/EIG/zstt22.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZSTT22 checks a set of M eigenvalues and eigenvectors,
 *
 *    A U = U S
 *
 * where A is Hermitian tridiagonal, the columns of U are unitary,
 * and S is diagonal (if KBAND=0) or Hermitian tridiagonal (if KBAND=1).
 * Two tests are performed:
 *
 *    RESULT(1) = | U* A U - S | / ( |A| m ulp )
 *
 *    RESULT(2) = | I - U*U | / ( m ulp )
 *
 * @param[in]     n       The size of the matrix. If zero, does nothing.
 * @param[in]     m       The number of eigenpairs to check. If zero, does nothing.
 * @param[in]     kband   The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     AD      Diagonal of A, dimension (n).
 * @param[in]     AE      Off-diagonal of A, dimension (n).
 * @param[in]     SD      Diagonal of S, dimension (n).
 * @param[in]     SE      Off-diagonal of S, dimension (n). Not referenced if kband=0.
 * @param[in]     U       The n by m unitary matrix U, dimension (ldu, m).
 * @param[in]     ldu     Leading dimension of U. ldu >= n.
 * @param[out]    work    Complex workspace array, dimension (ldwork, m+1).
 * @param[in]     ldwork  Leading dimension of work. ldwork >= max(1, m).
 * @param[out]    rwork   Real workspace array, dimension (n).
 * @param[out]    result  Array of dimension (2). The two test ratios.
 */
void zstt22(const INT n, const INT m, const INT kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const c128* const restrict U, const INT ldu,
            c128* const restrict work, const INT ldwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, j, k;
    f64 anorm, ulp, unfl, wnorm;
    c128 aukj;

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
            work[i + j * ldwork] = CZERO;
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

    wnorm = zlansy("1", "L", m, work, m, rwork);

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

    /* Compute  U'*U - I */
    /* NOTE: Reference LAPACK zstt22.f uses ZGEMM('T',...) here, not 'C'.
       This is looks like a bug in the reference, U^T*U instead of U^H*U  */
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, m, n, &CONE, U, ldu, U, ldu, &CZERO, work, m);

    for (j = 0; j < m; j++) {
        work[j + j * m] = work[j + j * m] - ONE;
    }

    result[1] = fmin((f64)m,
                     zlange("1", m, m, work, m, rwork)) / ((f64)m * ulp);
}
