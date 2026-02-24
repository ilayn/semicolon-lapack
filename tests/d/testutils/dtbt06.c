/**
 * @file dtbt06.c
 * @brief DTBT06 compares condition number estimates for a triangular band matrix.
 *
 * Port of LAPACK TESTING/LIN/dtbt06.f to C.
 */

#include <math.h>
#include "verify.h"

/* External declarations */
/**
 * DTBT06 computes a test ratio comparing RCOND (the reciprocal
 * condition number of a triangular matrix A) and RCONDC, the estimate
 * computed by DTBCON. Information about the triangular matrix A is
 * used if one estimate is zero and the other is non-zero to decide if
 * underflow in the estimate is justified.
 *
 * @param[in]     rcond   The estimate of the reciprocal condition number obtained by
 *                        forming the explicit inverse of the matrix A and computing
 *                        RCOND = 1/( norm(A) * norm(inv(A)) ).
 * @param[in]     rcondc  The estimate of the reciprocal condition number computed by
 *                        DTBCON.
 * @param[in]     uplo    Specifies whether the matrix A is upper or lower triangular.
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     diag    Specifies whether or not the matrix A is unit triangular.
 *                        = 'N': Non-unit triangular
 *                        = 'U': Unit triangular
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      The number of superdiagonals or subdiagonals of the
 *                        triangular band matrix A. kd >= 0.
 * @param[in]     AB      Array (ldab, n). The upper or lower triangular band matrix A,
 *                        stored in the first kd+1 rows of the array.
 * @param[in]     ldab    The leading dimension of the array AB. ldab >= kd+1.
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    rat     The test ratio. If both RCOND and RCONDC are nonzero,
 *                        RAT = MAX( RCOND, RCONDC )/MIN( RCOND, RCONDC ) - 1.
 *                        If RAT = 0, the two estimates are exactly the same.
 */
void dtbt06(const f64 rcond, const f64 rcondc,
            const char* uplo, const char* diag, const INT n, const INT kd,
            const f64* AB, const INT ldab, f64* work, f64* rat)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    f64 anorm, bignum, eps, rmax, rmin, smlnum;

    eps = dlamch("E");
    rmax = fmax(rcond, rcondc);
    rmin = fmin(rcond, rcondc);

    /* Do the easy cases first. */
    if (rmin < ZERO) {
        /* Invalid value for RCOND or RCONDC, return 1/EPS. */
        *rat = ONE / eps;
    } else if (rmin > ZERO) {
        /* Both estimates are positive, return RMAX/RMIN - 1. */
        *rat = rmax / rmin - ONE;
    } else if (rmax == ZERO) {
        /* Both estimates zero. */
        *rat = ZERO;
    } else {
        /* One estimate is zero, the other is non-zero. If the matrix is
         * ill-conditioned, return the nonzero estimate multiplied by
         * 1/EPS; if the matrix is badly scaled, return the nonzero
         * estimate multiplied by BIGNUM/TMAX, where TMAX is the maximum
         * element in absolute value in A. */
        smlnum = dlamch("S");
        bignum = ONE / smlnum;
        anorm = dlantb("M", uplo, diag, n, kd, AB, ldab, work);

        *rat = rmax * fmin(bignum / fmax(ONE, anorm), ONE / eps);
    }
}
