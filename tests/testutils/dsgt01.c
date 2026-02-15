/**
 * @file dsgt01.c
 * @brief DSGT01 checks a decomposition of the form A*Z = B*Z*D, A*B*Z = Z*D,
 *        or B*A*Z = Z*D.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

extern f64 dlamch(const char* cmach);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* A, const int lda, f64* work);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);

/**
 * DSGT01 checks a decomposition of the form
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
void dsgt01(const int itype, const char* uplo, const int n, const int m,
            const f64* A, const int lda,
            const f64* B, const int ldb,
            f64* Z, const int ldz,
            const f64* D, f64* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i;
    f64 anorm, ulp;
    CBLAS_UPLO cblas_uplo;

    result[0] = ZERO;
    if (n <= 0)
        return;

    ulp = dlamch("E");

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    /* Compute product of 1-norms of A and Z. */

    anorm = dlansy("1", uplo, n, A, lda, work) *
            dlange("1", n, m, Z, ldz, work);
    if (anorm == ZERO)
        anorm = ONE;

    if (itype == 1) {

        /* Norm of AZ - BZD */

        cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, A, lda, Z, ldz, ZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_dscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, B, ldb, Z, ldz, -ONE, work, n);

        result[0] = (dlange("1", n, m, work, n, work) / anorm) /
                    (n * ulp);

    } else if (itype == 2) {

        /* Norm of ABZ - ZD */

        cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, B, ldb, Z, ldz, ZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_dscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, A, lda, work, n, -ONE, Z, ldz);

        result[0] = (dlange("1", n, m, Z, ldz, work) / anorm) /
                    (n * ulp);

    } else if (itype == 3) {

        /* Norm of BAZ - ZD */

        cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, A, lda, Z, ldz, ZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_dscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, ONE, B, ldb, work, n, -ONE, Z, ldz);

        result[0] = (dlange("1", n, m, Z, ldz, work) / anorm) /
                    (n * ulp);
    }
}
