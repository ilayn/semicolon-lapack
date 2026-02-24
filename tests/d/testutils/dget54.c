/**
 * @file dget54.c
 * @brief DGET54 checks a generalized decomposition of the form A = U*S*V' and B = U*T*V'.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * DGET54 checks a generalized decomposition of the form
 *
 *          A = U*S*V'  and B = U*T* V'
 *
 * where ' means transpose and U and V are orthogonal.
 *
 * Specifically,
 *
 *  RESULT = ||( A - U*S*V', B - U*T*V' )|| / (||( A, B )||*n*ulp )
 *
 * @param[in]     n       The size of the matrix. If it is zero, DGET54 does nothing.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original (unfactored) matrix A.
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     B       Double precision array, dimension (ldb, n).
 *                        The original (unfactored) matrix B.
 * @param[in]     ldb     The leading dimension of B.
 * @param[in]     S       Double precision array, dimension (lds, n).
 *                        The factored matrix S.
 * @param[in]     lds     The leading dimension of S.
 * @param[in]     T       Double precision array, dimension (ldt, n).
 *                        The factored matrix T.
 * @param[in]     ldt     The leading dimension of T.
 * @param[in]     U       Double precision array, dimension (ldu, n).
 *                        The orthogonal matrix on the left-hand side.
 * @param[in]     ldu     The leading dimension of U.
 * @param[in]     V       Double precision array, dimension (ldv, n).
 *                        The orthogonal matrix on the right-hand side.
 * @param[in]     ldv     The leading dimension of V.
 * @param[out]    work    Double precision array, dimension (3*n*n).
 * @param[out]    result  The value RESULT. Limited to 1/ulp to avoid overflow.
 *                        Errors are flagged by RESULT=10/ulp.
 */
void dget54(const INT n, const f64* A, const INT lda,
            const f64* B, const INT ldb,
            const f64* S, const INT lds,
            const f64* T, const INT ldt,
            const f64* U, const INT ldu,
            const f64* V, const INT ldv,
            f64* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *result = ZERO;
    if (n <= 0)
        return;

    /* Constants */

    f64 unfl = dlamch("Safe minimum");
    f64 ulp = dlamch("Epsilon") * dlamch("Base");

    /* compute the norm of (A,B) */

    dlacpy("Full", n, n, A, lda, work, n);
    dlacpy("Full", n, n, B, ldb, work + (size_t)n * n, n);
    f64 abnorm = dlange("1", n, 2 * n, work, n, NULL);
    if (abnorm < unfl) abnorm = unfl;

    /* Compute W1 = A - U*S*V', and put in the array WORK(0:N*N-1) */

    dlacpy(" ", n, n, A, lda, work, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, ONE, U, ldu, S, lds, ZERO,
                work + (size_t)n * n, n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, -ONE, work + (size_t)n * n, n, V, ldv,
                ONE, work, n);

    /* Compute W2 = B - U*T*V', and put in WORK(N*N:2*N*N-1) */

    dlacpy(" ", n, n, B, ldb, work + (size_t)n * n, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, ONE, U, ldu, T, ldt, ZERO,
                work + 2 * (size_t)n * n, n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, -ONE, work + 2 * (size_t)n * n, n, V, ldv,
                ONE, work + (size_t)n * n, n);

    /* Compute norm(W)/ ( ulp*norm((A,B)) ) */

    f64 wnorm = dlange("1", n, 2 * n, work, n, NULL);

    if (abnorm > wnorm) {
        *result = (wnorm / abnorm) / (2 * n * ulp);
    } else {
        if (abnorm < ONE) {
            f64 tmp = wnorm;
            if (tmp > 2 * n * abnorm) tmp = 2 * n * abnorm;
            *result = (tmp / abnorm) / (2 * n * ulp);
        } else {
            f64 tmp = wnorm / abnorm;
            if (tmp > (f64)(2 * n)) tmp = (f64)(2 * n);
            *result = tmp / (2 * n * ulp);
        }
    }
}
