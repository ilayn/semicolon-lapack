/**
 * @file disnan.c
 * @brief DISNAN tests input for NaN.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DISNAN returns nonzero if its argument is NaN, and zero otherwise.
 *
 * @param[in] din  Input to test for NaN.
 *
 * @return Nonzero if din is NaN, zero otherwise.
 */
int disnan(const f64 din)
{
    return isnan(din);
}
