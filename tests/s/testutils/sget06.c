/**
 * @file sget06.c
 * @brief SGET06 computes a test ratio to compare two values for RCOND.
 */

#include <math.h>
#include "verify.h"

// Forward declaration
extern f32 slamch(const char* cmach);

/**
 * SGET06 computes a test ratio to compare two values for RCOND.
 *
 * The test ratio is computed as:
 *   if both RCOND and RCONDC are positive:
 *       ratio = max(RCOND, RCONDC) / min(RCOND, RCONDC) - (1 - EPS)
 *   otherwise:
 *       ratio = max(RCOND, RCONDC) / EPS
 *
 * A good condition number estimate should give a ratio close to 1.
 * The subtraction of (1-EPS) means that if RCOND == RCONDC, the ratio is ~EPS,
 * which is essentially 0 for practical purposes.
 *
 * @param[in] rcond   The estimate of the reciprocal of the condition number
 *                    of A, as computed by SGECON.
 * @param[in] rcondc  The reciprocal of the condition number of A, computed as
 *                    (1/norm(A)) / norm(inv(A)).
 *
 * @return The test ratio.
 */
f32 sget06(const f32 rcond, const f32 rcondc)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    f32 eps = slamch("E");
    f32 rat;

    if (rcond > ZERO) {
        if (rcondc > ZERO) {
            // Both positive: compute ratio of larger to smaller, subtract (1-eps)
            f32 maxval = (rcond > rcondc) ? rcond : rcondc;
            f32 minval = (rcond < rcondc) ? rcond : rcondc;
            rat = maxval / minval - (ONE - eps);
        } else {
            // rcond > 0, rcondc <= 0
            rat = rcond / eps;
        }
    } else {
        if (rcondc > ZERO) {
            // rcond <= 0, rcondc > 0
            rat = rcondc / eps;
        } else {
            // Both <= 0
            rat = ZERO;
        }
    }

    return rat;
}
