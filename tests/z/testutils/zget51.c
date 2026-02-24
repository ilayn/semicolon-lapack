/**
 * @file zget51.c
 * @brief ZGET51 checks a decomposition of the form A = U B V**H.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * ZGET51 generally checks a decomposition of the form
 *
 *      A = U B V**H
 *
 * where **H means conjugate transpose and U and V are unitary.
 *
 * Specifically, if ITYPE=1
 *
 *      RESULT = | A - U B V**H | / ( |A| n ulp )
 *
 * If ITYPE=2, then:
 *
 *      RESULT = | A - B | / ( |A| n ulp )
 *
 * If ITYPE=3, then:
 *
 *      RESULT = | I - U U**H | / ( n ulp )
 *
 * @param[in]     itype   Specifies the type of test.
 * @param[in]     n       The size of the matrix. n >= 0.
 * @param[in]     A       The original matrix, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     B       The factored matrix, dimension (ldb, n).
 * @param[in]     ldb     The leading dimension of B.
 * @param[in]     U       Left unitary matrix, dimension (ldu, n).
 * @param[in]     ldu     The leading dimension of U.
 * @param[in]     V       Right unitary matrix, dimension (ldv, n).
 * @param[in]     ldv     The leading dimension of V.
 * @param[out]    work    Complex workspace, dimension (2*n*n).
 * @param[out]    rwork   Real workspace, dimension (n).
 * @param[out]    result  The computed test value.
 */
void zget51(const INT itype, const INT n,
            const c128* A, const INT lda,
            const c128* B, const INT ldb,
            const c128* U, const INT ldu,
            const c128* V, const INT ldv,
            c128* work, f64* rwork, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEGCONE = CMPLX(-1.0, 0.0);

    *result = ZERO;
    if (n <= 0)
        return;

    f64 unfl = dlamch("Safe minimum");
    f64 ulp = dlamch("Epsilon") * dlamch("Base");

    if (itype < 1 || itype > 3) {
        *result = TEN / ulp;
        return;
    }

    if (itype <= 2) {
        f64 anorm = fmax(zlange("1", n, n, A, lda, rwork), unfl);

        if (itype == 1) {
            /* ITYPE=1: Compute W = A - U B V**H */
            zlacpy(" ", n, n, A, lda, work, n);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, &CONE, U, ldu, B, ldb, &CZERO,
                        &work[n * n], n);

            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        n, n, n, &NEGCONE, &work[n * n], n, V, ldv,
                        &CONE, work, n);
        } else {
            /* ITYPE=2: Compute W = A - B */
            zlacpy(" ", n, n, B, ldb, work, n);

            for (INT jcol = 0; jcol < n; jcol++) {
                for (INT jrow = 0; jrow < n; jrow++) {
                    work[jrow + n * jcol] = work[jrow + n * jcol]
                                            - A[jrow + lda * jcol];
                }
            }
        }

        f64 wnorm = zlange("1", n, n, work, n, rwork);

        if (anorm > wnorm) {
            *result = (wnorm / anorm) / (n * ulp);
        } else {
            if (anorm < ONE) {
                *result = (fmin(wnorm, n * anorm) / anorm) / (n * ulp);
            } else {
                *result = fmin(wnorm / anorm, (f64)n) / (n * ulp);
            }
        }
    } else {
        /* ITYPE=3: Compute U U**H - I */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

        for (INT jdiag = 0; jdiag < n; jdiag++) {
            work[(n + 1) * jdiag] = work[(n + 1) * jdiag] - CONE;
        }

        *result = fmin(zlange("1", n, n, work, n, rwork),
                       (f64)n) / (n * ulp);
    }
}
