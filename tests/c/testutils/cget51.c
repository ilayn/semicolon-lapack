/**
 * @file cget51.c
 * @brief CGET51 checks a decomposition of the form A = U B V**H.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * CGET51 generally checks a decomposition of the form
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
void cget51(const INT itype, const INT n,
            const c64* A, const INT lda,
            const c64* B, const INT ldb,
            const c64* U, const INT ldu,
            const c64* V, const INT ldv,
            c64* work, f32* rwork, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TEN = 10.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 NEGCONE = CMPLXF(-1.0f, 0.0f);

    *result = ZERO;
    if (n <= 0)
        return;

    f32 unfl = slamch("Safe minimum");
    f32 ulp = slamch("Epsilon") * slamch("Base");

    if (itype < 1 || itype > 3) {
        *result = TEN / ulp;
        return;
    }

    if (itype <= 2) {
        f32 anorm = fmaxf(clange("1", n, n, A, lda, rwork), unfl);

        if (itype == 1) {
            /* ITYPE=1: Compute W = A - U B V**H */
            clacpy(" ", n, n, A, lda, work, n);
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, &CONE, U, ldu, B, ldb, &CZERO,
                        &work[n * n], n);

            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        n, n, n, &NEGCONE, &work[n * n], n, V, ldv,
                        &CONE, work, n);
        } else {
            /* ITYPE=2: Compute W = A - B */
            clacpy(" ", n, n, B, ldb, work, n);

            for (INT jcol = 0; jcol < n; jcol++) {
                for (INT jrow = 0; jrow < n; jrow++) {
                    work[jrow + n * jcol] = work[jrow + n * jcol]
                                            - A[jrow + lda * jcol];
                }
            }
        }

        f32 wnorm = clange("1", n, n, work, n, rwork);

        if (anorm > wnorm) {
            *result = (wnorm / anorm) / (n * ulp);
        } else {
            if (anorm < ONE) {
                *result = (fminf(wnorm, n * anorm) / anorm) / (n * ulp);
            } else {
                *result = fminf(wnorm / anorm, (f32)n) / (n * ulp);
            }
        }
    } else {
        /* ITYPE=3: Compute U U**H - I */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CONE, U, ldu, U, ldu, &CZERO, work, n);

        for (INT jdiag = 0; jdiag < n; jdiag++) {
            work[(n + 1) * jdiag] = work[(n + 1) * jdiag] - CONE;
        }

        *result = fminf(clange("1", n, n, work, n, rwork),
                       (f32)n) / (n * ulp);
    }
}
