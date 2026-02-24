/**
 * @file sget51.c
 * @brief SGET51 checks a decomposition of the form A = U B V'.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * SGET51 generally checks a decomposition of the form
 *
 *      A = U B V'
 *
 * where ' means transpose and U and V are orthogonal.
 *
 * Specifically, if ITYPE=1
 *
 *      RESULT = | A - U B V' | / ( |A| n ulp )
 *
 * If ITYPE=2, then:
 *
 *      RESULT = | A - B | / ( |A| n ulp )
 *
 * If ITYPE=3, then:
 *
 *      RESULT = | I - UU' | / ( n ulp )
 *
 * @param[in]     itype   Specifies the type of test.
 * @param[in]     n       The size of the matrix. n >= 0.
 * @param[in]     A       The original matrix, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     B       The factored matrix, dimension (ldb, n).
 * @param[in]     ldb     The leading dimension of B.
 * @param[in]     U       Left orthogonal matrix, dimension (ldu, n).
 * @param[in]     ldu     The leading dimension of U.
 * @param[in]     V       Right orthogonal matrix, dimension (ldv, n).
 * @param[in]     ldv     The leading dimension of V.
 * @param[out]    work    Workspace, dimension (2*n*n).
 * @param[out]    result  The computed test value.
 */
void sget51(const INT itype, const INT n,
            const f32* A, const INT lda,
            const f32* B, const INT ldb,
            const f32* U, const INT ldu,
            const f32* V, const INT ldv,
            f32* work, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TEN = 10.0f;

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
        f32 anorm = fmaxf(slange("1", n, n, A, lda, work), unfl);

        if (itype == 1) {
            /* ITYPE=1: Compute W = A - UBV' */
            slacpy(" ", n, n, A, lda, work, n);
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, ONE, U, ldu, B, ldb, ZERO,
                        &work[n * n], n);

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        n, n, n, -ONE, &work[n * n], n, V, ldv,
                        ONE, work, n);
        } else {
            /* ITYPE=2: Compute W = A - B */
            slacpy(" ", n, n, B, ldb, work, n);

            for (INT jcol = 0; jcol < n; jcol++) {
                for (INT jrow = 0; jrow < n; jrow++) {
                    work[jrow + n * jcol] = work[jrow + n * jcol]
                                            - A[jrow + lda * jcol];
                }
            }
        }

        f32 wnorm = slange("1", n, n, work, n, &work[n * n]);

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
        /* ITYPE=3: Compute UU' - I */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

        for (INT jdiag = 0; jdiag < n; jdiag++) {
            work[(n + 1) * jdiag] = work[(n + 1) * jdiag] - ONE;
        }

        *result = fminf(slange("1", n, n, work, n, &work[n * n]),
                       (f32)n) / (n * ulp);
    }
}
