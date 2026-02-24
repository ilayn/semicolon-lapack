/**
 * @file zhet22.c
 * @brief ZHET22 checks a partial Hermitian eigendecomposition of the form
 *        A U = U S where A is Hermitian, the columns of U are orthonormal,
 *        and S is diagonal or Hermitian tridiagonal.
 *
 * Port of LAPACK's TESTING/EIG/zhet22.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zhet22(const INT itype, const char* uplo, const INT n, const INT m,
            const INT kband, const c128* const restrict A, const INT lda,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict U, const INT ldu,
            const c128* const restrict V, const INT ldv,
            const c128* const restrict tau,
            c128* const restrict work, f64* const restrict rwork,
            f64* restrict result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT j, jj, jj1, jj2, nn, nnp1;
    f64 anorm, ulp, unfl, wnorm;

    (void)V;
    (void)ldv;
    (void)tau;

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0 || m <= 0)
        return;

    unfl = dlamch("S");
    ulp = dlamch("P");

    /* Norm of A */
    anorm = zlanhe("1", uplo, n, A, lda, rwork);
    if (anorm < unfl)
        anorm = unfl;

    /*
     * Compute error matrix:
     * ITYPE=1: error = U**H A U - S
     *
     * First compute WORK = A * U using ZHEMM
     */
    cblas_zhemm(CblasColMajor, CblasLeft,
                (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower,
                n, m, &CONE, A, lda, U, ldu, &CZERO, work, n);

    nn = n * n;
    nnp1 = nn;

    /* Compute WORK(nnp1:) = U**H * WORK = U**H * A * U */
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, m, n, &CONE, U, ldu, work, n, &CZERO, &work[nnp1], n);

    /* Subtract diagonal D from the result */
    for (j = 0; j < m; j++) {
        jj = nnp1 + j * n + j;
        work[jj] = work[jj] - D[j];
    }

    /* Subtract off-diagonal E if kband=1 */
    if (kband == 1 && n > 1) {
        for (j = 1; j < m; j++) {
            jj1 = nnp1 + j * n + j - 1;
            jj2 = nnp1 + (j - 1) * n + j;
            work[jj1] = work[jj1] - E[j - 1];
            work[jj2] = work[jj2] - E[j - 1];
        }
    }

    /* Compute norm of U**H A U - S */
    wnorm = zlanhe("1", uplo, m, &work[nnp1], n, rwork);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (m * ulp);
    } else {
        if (anorm < ONE) {
            f64 tmp = fmin(wnorm, (f64)m * anorm);
            result[0] = (tmp / anorm) / (m * ulp);
        } else {
            f64 tmp = fmin(wnorm / anorm, (f64)m);
            result[0] = tmp / (m * ulp);
        }
    }

    /*
     * Test 2: Compute U**H U - I (only for itype=1)
     */
    if (itype == 1) {
        zunt01("C", n, m, U, ldu, work, 2 * n * n, rwork, &result[1]);
    }
}
