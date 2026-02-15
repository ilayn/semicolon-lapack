/**
 * @file cgecon.c
 * @brief Estimates the reciprocal of the condition number of a complex general matrix.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGECON estimates the reciprocal of the condition number of a general
 * complex matrix A, in either the 1-norm or the infinity-norm, using
 * the LU factorization computed by CGETRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as
 *    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 * @param[in]     norm  Specifies whether the 1-norm condition number or the
 *                      infinity-norm condition number is required:
 *                      - '1' or 'O': 1-norm
 *                      - 'I': Infinity-norm
 * @param[in]     n     The order of the matrix A (n >= 0).
 * @param[in]     A     The factors L and U from the factorization A = P*L*U
 *                      as computed by cgetrf. Complex array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,n)).
 * @param[in]     anorm If norm = '1' or "O", the 1-norm of the original matrix A.
 *                      If norm = "I", the infinity-norm of the original matrix A.
 * @param[out]    rcond The reciprocal of the condition number of the matrix A,
 *                      computed as RCOND = 1/(norm(A) * norm(inv(A))).
 * @param[out]    work  Complex workspace array of dimension (2*n).
 * @param[out]    rwork Real workspace array of dimension (2*n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value.
 *                           NaNs are illegal values for anorm, and they propagate
 *                           to the output parameter rcond.
 *                           Infinity is illegal for anorm, and it propagates to the
 *                           output parameter rcond as 0.
 *                           - = 1: if rcond = NaN, or rcond = Inf, or the computed norm
 *                           of the inverse of A is 0. In the latter, rcond = 0.
 */
void cgecon(
    const char* norm,
    const int n,
    const c64* restrict A,
    const int lda,
    const f32 anorm,
    f32* rcond,
    c64* restrict work,
    f32* restrict rwork,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    int onenrm;
    char normin;
    int ix, kase, kase1;
    f32 ainvnm, scale, sl, smlnum, su, hugeval;
    int isave[3];
    int linfo;

    hugeval = FLT_MAX;

    // Test the input parameters
    *info = 0;
    onenrm = (norm[0] == '1' || norm[0] == 'O' || norm[0] == 'o');
    if (!onenrm && norm[0] != 'I' && norm[0] != 'i') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (anorm < ZERO) {
        *info = -5;
    }

    if (*info != 0) {
        xerbla("CGECON", -(*info));
        return;
    }

    // Quick return if possible
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    } else if (isnan(anorm)) {
        *rcond = anorm;
        *info = -5;
        return;
    } else if (anorm > hugeval) {
        *info = -5;
        return;
    }

    smlnum = FLT_MIN;

    // Estimate the norm of inv(A)
    ainvnm = ZERO;
    normin = 'N';
    if (onenrm) {
        kase1 = 1;
    } else {
        kase1 = 2;
    }
    kase = 0;

    while (1) {
        // work[0:n-1] = X, work[n:2n-1] = V
        clacn2(n, &work[n], work, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (kase == kase1) {
            // Multiply by inv(L)
            clatrs("L", "N", "U", &normin, n, A, lda, work, &sl, rwork, &linfo);

            // Multiply by inv(U)
            clatrs("U", "N", "N", &normin, n, A, lda, work, &su, &rwork[n], &linfo);
        } else {
            // Multiply by inv(U**H)
            clatrs("U", "C", "N", &normin, n, A, lda, work, &su, &rwork[n], &linfo);

            // Multiply by inv(L**H)
            clatrs("L", "C", "U", &normin, n, A, lda, work, &sl, rwork, &linfo);
        }

        // Divide X by 1/(SL*SU) if doing so will not cause overflow
        scale = sl * su;
        normin = 'Y';
        if (scale != ONE) {
            ix = cblas_icamax(n, work, 1);
            if (scale < (fabsf(crealf(work[ix])) + fabsf(cimagf(work[ix]))) * smlnum || scale == ZERO) {
                return;
            }
            cdrscl(n, scale, work, 1);
        }
    }

    // Compute the estimate of the reciprocal condition number
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    } else {
        *info = 1;
        return;
    }

    // Check for NaNs and Infs
    if (isnan(*rcond) || *rcond > hugeval) {
        *info = 1;
    }
}
