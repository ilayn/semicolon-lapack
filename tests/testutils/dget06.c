/**
 * @file dget06.c
 * @brief DGET06 computes a test ratio to compare two values for RCOND.
 */

#include <math.h>
#include "verify.h"

// Forward declaration
extern double dlamch(const char* cmach);

/**
 * DGET06 computes a test ratio to compare two values for RCOND.
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
 *                    of A, as computed by DGECON.
 * @param[in] rcondc  The reciprocal of the condition number of A, computed as
 *                    (1/norm(A)) / norm(inv(A)).
 *
 * @return The test ratio.
 */
double dget06(const double rcond, const double rcondc)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    double eps = dlamch("E");
    double rat;

    if (rcond > ZERO) {
        if (rcondc > ZERO) {
            // Both positive: compute ratio of larger to smaller, subtract (1-eps)
            double maxval = (rcond > rcondc) ? rcond : rcondc;
            double minval = (rcond < rcondc) ? rcond : rcondc;
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
