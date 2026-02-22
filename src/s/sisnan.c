/**
 * @file sisnan.c
 * @brief SISNAN tests input for NaN.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SISNAN returns nonzero if its argument is NaN, and zero otherwise.
 *
 * @param[in] din  Input to test for NaN.
 *
 * @return Nonzero if din is NaN, zero otherwise.
 */
INT sisnan(const f32 din)
{
    return isnan(din);
}
