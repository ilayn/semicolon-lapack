/**
 * @file dget51.c
 * @brief DGET51 checks a decomposition of the form A = U B V'.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                  const f64* A, const int lda, f64* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);

/**
 * DGET51 generally checks a decomposition of the form
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
void dget51(const int itype, const int n,
            const f64* A, const int lda,
            const f64* B, const int ldb,
            const f64* U, const int ldu,
            const f64* V, const int ldv,
            f64* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TEN = 10.0;

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
        f64 anorm = fmax(dlange("1", n, n, A, lda, work), unfl);

        if (itype == 1) {
            /* ITYPE=1: Compute W = A - UBV' */
            dlacpy(" ", n, n, A, lda, work, n);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, ONE, U, ldu, B, ldb, ZERO,
                        &work[n * n], n);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        n, n, n, -ONE, &work[n * n], n, V, ldv,
                        ONE, work, n);
        } else {
            /* ITYPE=2: Compute W = A - B */
            dlacpy(" ", n, n, B, ldb, work, n);

            for (int jcol = 0; jcol < n; jcol++) {
                for (int jrow = 0; jrow < n; jrow++) {
                    work[jrow + n * jcol] = work[jrow + n * jcol]
                                            - A[jrow + lda * jcol];
                }
            }
        }

        f64 wnorm = dlange("1", n, n, work, n, &work[n * n]);

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
        /* ITYPE=3: Compute UU' - I */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

        for (int jdiag = 0; jdiag < n; jdiag++) {
            work[(n + 1) * jdiag] = work[(n + 1) * jdiag] - ONE;
        }

        *result = fmin(dlange("1", n, n, work, n, &work[n * n]),
                       (f64)n) / (n * ulp);
    }
}
