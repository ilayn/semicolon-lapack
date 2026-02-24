/**
 * @file zget54.c
 * @brief ZGET54 checks a generalized decomposition of the form A = U*S*V' and B = U*T*V'.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * ZGET54 checks a generalized decomposition of the form
 *
 *          A = U*S*V'  and B = U*T* V'
 *
 * where ' means conjugate transpose and U and V are unitary.
 *
 * Specifically,
 *
 *  RESULT = ||( A - U*S*V', B - U*T*V' )|| / (||( A, B )||*n*ulp )
 *
 * @param[in]     n       The size of the matrix. If it is zero, ZGET54 does nothing.
 * @param[in]     A       Complex*16 array, dimension (lda, n).
 *                        The original (unfactored) matrix A.
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     B       Complex*16 array, dimension (ldb, n).
 *                        The original (unfactored) matrix B.
 * @param[in]     ldb     The leading dimension of B.
 * @param[in]     S       Complex*16 array, dimension (lds, n).
 *                        The factored matrix S.
 * @param[in]     lds     The leading dimension of S.
 * @param[in]     T       Complex*16 array, dimension (ldt, n).
 *                        The factored matrix T.
 * @param[in]     ldt     The leading dimension of T.
 * @param[in]     U       Complex*16 array, dimension (ldu, n).
 *                        The orthogonal matrix on the left-hand side.
 * @param[in]     ldu     The leading dimension of U.
 * @param[in]     V       Complex*16 array, dimension (ldv, n).
 *                        The orthogonal matrix on the right-hand side.
 * @param[in]     ldv     The leading dimension of V.
 * @param[out]    work    Complex*16 array, dimension (3*n*n).
 * @param[out]    result  The value RESULT. Limited to 1/ulp to avoid overflow.
 *                        Errors are flagged by RESULT=10/ulp.
 */
void zget54(const INT n, const c128* A, const INT lda,
            const c128* B, const INT ldb,
            const c128* S, const INT lds,
            const c128* T, const INT ldt,
            const c128* U, const INT ldu,
            const c128* V, const INT ldv,
            c128* work, f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CMONE = CMPLX(-1.0, 0.0);

    *result = ZERO;
    if (n <= 0)
        return;

    /* Constants */

    f64 unfl = dlamch("Safe minimum");
    f64 ulp = dlamch("Epsilon") * dlamch("Base");

    /* compute the norm of (A,B) */

    zlacpy("Full", n, n, A, lda, work, n);
    zlacpy("Full", n, n, B, ldb, work + (size_t)n * n, n);
    f64 abnorm = zlange("1", n, 2 * n, work, n, NULL);
    if (abnorm < unfl) abnorm = unfl;

    /* Compute W1 = A - U*S*V', and put in the array WORK(0:N*N-1) */

    zlacpy(" ", n, n, A, lda, work, n);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, &CONE, U, ldu, S, lds, &CZERO,
                work + (size_t)n * n, n);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, n, n, &CMONE, work + (size_t)n * n, n, V, ldv,
                &CONE, work, n);

    /* Compute W2 = B - U*T*V', and put in WORK(N*N:2*N*N-1) */

    zlacpy(" ", n, n, B, ldb, work + (size_t)n * n, n);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, &CONE, U, ldu, T, ldt, &CZERO,
                work + 2 * (size_t)n * n, n);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, n, n, &CMONE, work + 2 * (size_t)n * n, n, V, ldv,
                &CONE, work + (size_t)n * n, n);

    /* Compute norm(W)/ ( ulp*norm((A,B)) ) */

    f64 wnorm = zlange("1", n, 2 * n, work, n, NULL);

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
