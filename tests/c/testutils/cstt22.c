/**
 * @file cstt22.c
 * @brief CSTT22 checks a set of M eigenvalues and eigenvectors
 *        of a Hermitian tridiagonal matrix.
 *
 * Port of LAPACK's TESTING/EIG/cstt22.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CSTT22 checks a set of M eigenvalues and eigenvectors,
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
void cstt22(const INT n, const INT m, const INT kband,
            const f32* const restrict AD, const f32* const restrict AE,
            const f32* const restrict SD, const f32* const restrict SE,
            const c64* const restrict U, const INT ldu,
            c64* const restrict work, const INT ldwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j, k;
    f32 anorm, ulp, unfl, wnorm;
    c64 aukj;

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0 || m <= 0)
        return;

    unfl = slamch("S");
    ulp = slamch("E");

    /* Do Test 1 */

    /* Compute the 1-norm of A. */
    if (n > 1) {
        anorm = fabsf(AD[0]) + fabsf(AE[0]);
        for (j = 1; j < n - 1; j++) {
            anorm = fmaxf(anorm, fabsf(AD[j]) + fabsf(AE[j]) + fabsf(AE[j - 1]));
        }
        anorm = fmaxf(anorm, fabsf(AD[n - 1]) + fabsf(AE[n - 2]));
    } else {
        anorm = fabsf(AD[0]);
    }
    anorm = fmaxf(anorm, unfl);

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

    wnorm = clansy("1", "L", m, work, m, rwork);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / ((f32)m * ulp);
    } else {
        if (anorm < ONE) {
            result[0] = (fminf(wnorm, (f32)m * anorm) / anorm) / ((f32)m * ulp);
        } else {
            result[0] = fminf(wnorm / anorm, (f32)m) / ((f32)m * ulp);
        }
    }

    /* Do Test 2 */

    /* Compute  U'*U - I */
    /* NOTE: Reference LAPACK cstt22.f uses ZGEMM('T',...) here, not 'C'.
       This is looks like a bug in the reference, U^T*U instead of U^H*U  */
    cblas_cgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, m, n, &CONE, U, ldu, U, ldu, &CZERO, work, m);

    for (j = 0; j < m; j++) {
        work[j + j * m] = work[j + j * m] - ONE;
    }

    result[1] = fminf((f32)m,
                     clange("1", m, m, work, m, rwork)) / ((f32)m * ulp);
}
