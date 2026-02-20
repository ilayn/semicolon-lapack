/**
 * @file stpt06.c
 * @brief STPT06 compares condition number estimates for a triangular matrix
 *        in packed format.
 *
 * Port of LAPACK TESTING/LIN/stpt06.f to C.
 */

#include <math.h>
#include "verify.h"

/* External declarations */
extern f32 slamch(const char* cmach);
extern f32 slantp(const char* norm, const char* uplo, const char* diag,
                     const int n, const f32* AP, f32* work);

/**
 * STPT06 computes a test ratio comparing RCOND (the reciprocal
 * condition number of a triangular matrix A) and RCONDC, the estimate
 * computed by STPCON. Information about the triangular matrix A is
 * used if one estimate is zero and the other is non-zero to decide if
 * underflow in the estimate is justified.
 *
 * @param[in]     rcond   The estimate of the reciprocal condition number from
 *                        forming the explicit inverse: 1/(norm(A) * norm(inv(A))).
 * @param[in]     rcondc  The estimate of the reciprocal condition number from STPCON.
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     AP      Array (n*(n+1)/2). The triangular matrix A in packed storage.
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    rat     The test ratio:
 *                        If both estimates are nonzero: max(rcond, rcondc)/min(rcond, rcondc) - 1.
 *                        If both are zero: 0.
 *                        If one is zero: scaled version of the nonzero estimate.
 */
void stpt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const int n,
            const f32* AP, f32* work, f32* rat)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    f32 anorm, bignum, eps, rmax, rmin, smlnum;

    eps = slamch("E");
    rmax = fmaxf(rcond, rcondc);
    rmin = fminf(rcond, rcondc);

    /* Do the easy cases first */
    if (rmin < ZERO) {
        /* Invalid value for RCOND or RCONDC, return 1/EPS */
        *rat = ONE / eps;
    } else if (rmin > ZERO) {
        /* Both estimates are positive, return RMAX/RMIN - 1 */
        *rat = rmax / rmin - ONE;
    } else if (rmax == ZERO) {
        /* Both estimates zero */
        *rat = ZERO;
    } else {
        /* One estimate is zero, the other is non-zero. If the matrix is
         * ill-conditioned, return the nonzero estimate multiplied by
         * 1/EPS; if the matrix is badly scaled, return the nonzero
         * estimate multiplied by BIGNUM/TMAX, where TMAX is the maximum
         * element in absolute value in A. */
        smlnum = slamch("S");
        bignum = ONE / smlnum;
        anorm = slantp("M", uplo, diag, n, AP, work);

        *rat = rmax * fminf(bignum / fmaxf(ONE, anorm), ONE / eps);
    }
}
