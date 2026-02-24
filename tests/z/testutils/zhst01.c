/**
 * @file zhst01.c
 * @brief ZHST01 tests the reduction of a general matrix A to upper Hessenberg form.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZHST01 tests the reduction of a general matrix A to upper Hessenberg
 * form: A = Q*H*Q'. Two test ratios are computed:
 *
 *    RESULT[0] = norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
 *    RESULT[1] = norm( I - Q'*Q ) / ( N * EPS )
 *
 * The matrix Q is assumed to be given explicitly as it would be
 * following ZGEHRD + ZUNGHR.
 *
 * In this version, ILO and IHI are not used, but they could be used
 * to save some work if this is desired.
 *
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] ilo     Not used in this version.
 * @param[in] ihi     Not used in this version.
 * @param[in] A       The original n by n matrix A, dimension (lda, n).
 * @param[in] lda     The leading dimension of A. lda >= max(1, n).
 * @param[in] H       The upper Hessenberg matrix H from A = Q*H*Q',
 *                    dimension (ldh, n). H is assumed to be zero below
 *                    the first subdiagonal.
 * @param[in] ldh     The leading dimension of H. ldh >= max(1, n).
 * @param[in] Q       The unitary matrix Q from A = Q*H*Q',
 *                    dimension (ldq, n).
 * @param[in] ldq     The leading dimension of Q. ldq >= max(1, n).
 * @param[out] work   Workspace array, dimension (lwork).
 * @param[in] lwork   The length of work. lwork >= 2*n*n.
 * @param[out] rwork  Real workspace array, dimension (n).
 * @param[out] result Array of 2 elements containing the test ratios.
 */
void zhst01(const INT n, const INT ilo, const INT ihi,
            const c128* A, const INT lda,
            const c128* H, const INT ldh,
            const c128* Q, const INT ldq,
            c128* work, const INT lwork, f64* rwork, f64* result)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT ldwork;
    f64 anorm, eps, smlnum, unfl, wnorm;

    /* Silence unused parameter warnings */
    (void)ilo;
    (void)ihi;

    /* Quick return if possible */
    if (n <= 0) {
        result[0] = ZERO;
        result[1] = ZERO;
        return;
    }

    unfl = dlamch("S");
    eps = dlamch("P");
    smlnum = unfl * n / eps;

    /*
     * Test 1: Compute norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
     *
     * Copy A to WORK
     */
    ldwork = (1 > n) ? 1 : n;
    zlacpy(" ", n, n, A, lda, work, ldwork);

    /* Compute Q*H and store in WORK(n*ldwork:) */
    {
        const c128 CONE = CMPLX(ONE, 0.0);
        const c128 CZERO = CMPLX(ZERO, 0.0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, &CONE, Q, ldq, H, ldh, &CZERO,
                    &work[ldwork * n], ldwork);
    }

    /* Compute A - Q*H*Q' (result stored in WORK(0:n*ldwork-1)) */
    {
        const c128 CMONE = CMPLX(-ONE, 0.0);
        const c128 CONE = CMPLX(ONE, 0.0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CMONE, &work[ldwork * n], ldwork, Q, ldq,
                    &CONE, work, ldwork);
    }

    anorm = zlange("1", n, n, A, lda, rwork);
    if (anorm < unfl) {
        anorm = unfl;
    }
    wnorm = zlange("1", n, n, work, ldwork, rwork);

    /*
     * Note that RESULT[0] cannot overflow and is bounded by 1/(N*EPS)
     */
    {
        f64 num = (wnorm < anorm) ? wnorm : anorm;
        f64 denom = (smlnum > anorm * eps) ? smlnum : anorm * eps;
        result[0] = num / denom / n;
    }

    /*
     * Test 2: Compute norm( I - Q'*Q ) / ( N * EPS )
     */
    zunt01("C", n, n, Q, ldq, work, lwork, rwork, &result[1]);
}
