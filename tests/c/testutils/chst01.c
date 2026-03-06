/**
 * @file chst01.c
 * @brief CHST01 tests the reduction of a general matrix A to upper Hessenberg form.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CHST01 tests the reduction of a general matrix A to upper Hessenberg
 * form: A = Q*H*Q'. Two test ratios are computed:
 *
 *    RESULT[0] = norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
 *    RESULT[1] = norm( I - Q'*Q ) / ( N * EPS )
 *
 * The matrix Q is assumed to be given explicitly as it would be
 * following CGEHRD + CUNGHR.
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
void chst01(const INT n, const INT ilo, const INT ihi,
            const c64* A, const INT lda,
            const c64* H, const INT ldh,
            const c64* Q, const INT ldq,
            c64* work, const INT lwork, f32* rwork, f32* result)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT ldwork;
    f32 anorm, eps, smlnum, unfl, wnorm;

    /* Silence unused parameter warnings */
    (void)ilo;
    (void)ihi;

    /* Quick return if possible */
    if (n <= 0) {
        result[0] = ZERO;
        result[1] = ZERO;
        return;
    }

    unfl = slamch("S");
    eps = slamch("P");
    smlnum = unfl * n / eps;

    /*
     * Test 1: Compute norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
     *
     * Copy A to WORK
     */
    ldwork = (1 > n) ? 1 : n;
    clacpy(" ", n, n, A, lda, work, ldwork);

    /* Compute Q*H and store in WORK(n*ldwork:) */
    {
        const c64 CONE = CMPLXF(ONE, 0.0f);
        const c64 CZERO = CMPLXF(ZERO, 0.0f);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, &CONE, Q, ldq, H, ldh, &CZERO,
                    &work[ldwork * n], ldwork);
    }

    /* Compute A - Q*H*Q' (result stored in WORK(0:n*ldwork-1)) */
    {
        const c64 CMONE = CMPLXF(-ONE, 0.0f);
        const c64 CONE = CMPLXF(ONE, 0.0f);
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CMONE, &work[ldwork * n], ldwork, Q, ldq,
                    &CONE, work, ldwork);
    }

    anorm = clange("1", n, n, A, lda, rwork);
    if (anorm < unfl) {
        anorm = unfl;
    }
    wnorm = clange("1", n, n, work, ldwork, rwork);

    /*
     * Note that RESULT[0] cannot overflow and is bounded by 1/(N*EPS)
     */
    {
        f32 num = (wnorm < anorm) ? wnorm : anorm;
        f32 denom = (smlnum > anorm * eps) ? smlnum : anorm * eps;
        result[0] = num / denom / n;
    }

    /*
     * Test 2: Compute norm( I - Q'*Q ) / ( N * EPS )
     */
    cunt01("C", n, n, Q, ldq, work, lwork, rwork, &result[1]);
}
