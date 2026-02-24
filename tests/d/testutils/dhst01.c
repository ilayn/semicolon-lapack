/**
 * @file dhst01.c
 * @brief DHST01 tests the reduction of a general matrix A to upper Hessenberg form.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DHST01 tests the reduction of a general matrix A to upper Hessenberg
 * form: A = Q*H*Q'. Two test ratios are computed:
 *
 *    RESULT[0] = norm( A - Q*H*Q' ) / ( norm(A) * N * EPS )
 *    RESULT[1] = norm( I - Q'*Q ) / ( N * EPS )
 *
 * The matrix Q is assumed to be given explicitly as it would be
 * following DGEHRD + DORGHR.
 *
 * In this version, ILO and IHI are not used and are assumed to be 1 and
 * N, respectively.
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
 * @param[in] Q       The orthogonal matrix Q from A = Q*H*Q',
 *                    dimension (ldq, n).
 * @param[in] ldq     The leading dimension of Q. ldq >= max(1, n).
 * @param[out] work   Workspace array, dimension (lwork).
 * @param[in] lwork   The length of work. lwork >= 2*n*n.
 * @param[out] result Array of 2 elements containing the test ratios.
 */
void dhst01(const INT n, const INT ilo, const INT ihi,
            const f64* A, const INT lda,
            const f64* H, const INT ldh,
            const f64* Q, const INT ldq,
            f64* work, const INT lwork, f64* result)
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
    dlacpy(" ", n, n, A, lda, work, ldwork);

    /* Compute Q*H and store in WORK(n*ldwork:) */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, ONE, Q, ldq, H, ldh, ZERO,
                &work[ldwork * n], ldwork);

    /* Compute A - Q*H*Q' (result stored in WORK(0:n*ldwork-1)) */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, -ONE, &work[ldwork * n], ldwork, Q, ldq,
                ONE, work, ldwork);

    anorm = dlange("1", n, n, A, lda, &work[ldwork * n]);
    if (anorm < unfl) {
        anorm = unfl;
    }
    wnorm = dlange("1", n, n, work, ldwork, &work[ldwork * n]);

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
    dort01("C", n, n, Q, ldq, work, lwork, &result[1]);
}
