/**
 * @file dlaisnan.c
 * @brief DLAISNAN tests input for NaN by comparing two arguments for inequality.
 */

#include "semicolon_lapack_double.h"

/**
 * This routine is not for general use. It exists solely to avoid
 * over-optimization in DISNAN.
 *
 * DLAISNAN checks for NaNs by comparing its two arguments for
 * inequality. NaN is the only floating-point value where NaN != NaN
 * returns TRUE. To check for NaNs, pass the same variable as both
 * arguments.
 *
 * A compiler must assume that the two arguments are
 * not the same variable, and the test will not be optimized away.
 * Interprocedural or whole-program optimization may delete this
 * test. The ISNAN functions will be replaced by the correct
 * Fortran 03 intrinsic once the intrinsic is widely available.
 *
 * @param[in] din1
 *          First number to compare for inequality.
 *
 * @param[in] din2
 *          Second number to compare for inequality.
 *
 * @return 1 if din1 != din2 (true for NaN), 0 otherwise.
 */
INT dlaisnan(const f64 din1, const f64 din2)
{
    return (din1 != din2) ? 1 : 0;
}
