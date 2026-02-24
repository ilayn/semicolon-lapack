/**
 * @file ssgt01.c
 * @brief SSGT01 checks a decomposition of the form A*Z = B*Z*D, A*B*Z = Z*D,
 *        or B*A*Z = Z*D.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SSGT01 checks a decomposition of the form
 *
 *    A Z   = B Z D   or
 *    A B Z = Z D     or
 *    B A Z = Z D
 *
 * where A is a symmetric matrix, B is symmetric positive definite,
 * Z is orthogonal, and D is diagonal.
 *
 * One of the following test ratios is computed:
 *
 *    ITYPE = 1:  RESULT(1) = | A Z - B Z D | / ( |A| |Z| n ulp )
 *    ITYPE = 2:  RESULT(1) = | A B Z - Z D | / ( |A| |Z| n ulp )
 *    ITYPE = 3:  RESULT(1) = | B A Z - Z D | / ( |A| |Z| n ulp )
 *
 * @param[in]     itype  The form of the generalized eigenproblem (1, 2, or 3).
 * @param[in]     uplo   'U' or 'L' for upper/lower triangle storage.
 * @param[in]     n      The order of the matrix A.
 * @param[in]     m      The number of eigenvalues found. 0 <= m <= n.
 * @param[in]     A      The original symmetric matrix, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A.
 * @param[in]     B      The symmetric positive definite matrix, dimension (ldb, n).
 * @param[in]     ldb    Leading dimension of B.
 * @param[in,out] Z      The computed eigenvectors, dimension (ldz, m). Modified for itype 2,3.
 * @param[in]     ldz    Leading dimension of Z.
 * @param[in]     D      The computed eigenvalues, dimension (m).
 * @param[out]    work   Workspace, dimension (n*n).
 * @param[out]    result The test ratio.
 */
void ssgt01(const INT itype, const char* uplo, const INT n, const INT m,
            const f32* A, const INT lda,
            const f32* B, const INT ldb,
            f32* Z, const INT ldz,
            const f32* D, f32* work, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i;
    f32 anorm, ulp;
    CBLAS_UPLO cblas_uplo;

    result[0] = ZERO;
    if (n <= 0)
        return;

    ulp = slamch("E");

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    /* Compute product of 1-norms of A and Z. */

    anorm = slansy("1", uplo, n, A, lda, work) *
            slange("1", n, m, Z, ldz, work);
    if (anorm == ZERO)
        anorm = ONE;

    if (itype == 1) {

        /* Norm of AZ - BZD */

        cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, A, lda, Z, ldz, ZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_sscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, B, ldb, Z, ldz, -ONE, work, n);

        result[0] = (slange("1", n, m, work, n, &work[n * m]) / anorm) /
                    (n * ulp);

    } else if (itype == 2) {

        /* Norm of ABZ - ZD */

        cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, B, ldb, Z, ldz, ZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_sscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, A, lda, work, n, -ONE, Z, ldz);

        result[0] = (slange("1", n, m, Z, ldz, work) / anorm) /
                    (n * ulp);

    } else if (itype == 3) {

        /* Norm of BAZ - ZD */

        cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, A, lda, Z, ldz, ZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_sscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, B, ldb, work, n, -ONE, Z, ldz);

        result[0] = (slange("1", n, m, Z, ldz, work) / anorm) /
                    (n * ulp);
    }
}
