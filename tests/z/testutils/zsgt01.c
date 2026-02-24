/**
 * @file zsgt01.c
 * @brief ZSGT01 checks a decomposition of the form A*Z = B*Z*D, A*B*Z = Z*D,
 *        or B*A*Z = Z*D.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZSGT01 checks a decomposition of the form
 *
 *    A Z   = B Z D   or
 *    A B Z = Z D     or
 *    B A Z = Z D
 *
 * where A is a Hermitian matrix, B is Hermitian positive definite,
 * Z is unitary, and D is diagonal.
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
 * @param[in]     A      The original Hermitian matrix, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A.
 * @param[in]     B      The Hermitian positive definite matrix, dimension (ldb, n).
 * @param[in]     ldb    Leading dimension of B.
 * @param[in,out] Z      The computed eigenvectors, dimension (ldz, m). Modified for itype 2,3.
 * @param[in]     ldz    Leading dimension of Z.
 * @param[in]     D      The computed eigenvalues, dimension (m).
 * @param[out]    work   Complex workspace, dimension (n*n).
 * @param[out]    rwork  Real workspace, dimension (n).
 * @param[out]    result The test ratio.
 */
void zsgt01(const INT itype, const char* uplo, const INT n, const INT m,
            const c128* A, const INT lda,
            const c128* B, const INT ldb,
            c128* Z, const INT ldz,
            const f64* D, c128* work, f64* rwork, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CMONE = CMPLX(-1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT i;
    f64 anorm, ulp;
    CBLAS_UPLO cblas_uplo;

    result[0] = ZERO;
    if (n <= 0)
        return;

    ulp = dlamch("E");

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    /* Compute product of 1-norms of A and Z. */

    anorm = zlanhe("1", uplo, n, A, lda, rwork) *
            zlange("1", n, m, Z, ldz, rwork);
    if (anorm == ZERO)
        anorm = ONE;

    if (itype == 1) {

        /* Norm of AZ - BZD */

        cblas_zhemm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, &CONE, A, lda, Z, ldz, &CZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_zdscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_zhemm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, &CONE, B, ldb, Z, ldz, &CMONE, work, n);

        result[0] = (zlange("1", n, m, work, n, rwork) / anorm) /
                    (n * ulp);

    } else if (itype == 2) {

        /* Norm of ABZ - ZD */

        cblas_zhemm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, &CONE, B, ldb, Z, ldz, &CZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_zdscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_zhemm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, &CONE, A, lda, work, n, &CMONE, Z, ldz);

        result[0] = (zlange("1", n, m, Z, ldz, rwork) / anorm) /
                    (n * ulp);

    } else if (itype == 3) {

        /* Norm of BAZ - ZD */

        cblas_zhemm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, &CONE, A, lda, Z, ldz, &CZERO, work, n);
        for (i = 0; i < m; i++) {
            cblas_zdscal(n, D[i], &Z[i * ldz], 1);
        }
        cblas_zhemm(CblasColMajor, CblasLeft, cblas_uplo,
                    n, m, &CONE, B, ldb, work, n, &CMONE, Z, ldz);

        result[0] = (zlange("1", n, m, Z, ldz, rwork) / anorm) /
                    (n * ulp);
    }
}
