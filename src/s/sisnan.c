/**
 * @file sisnan.c
 * @brief SISNAN tests input for NaN.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SISNAN returns nonzero if its argument is NaN, and zero otherwise.
 *
 * @param[in] sin  Input to test for NaN.
 *
 * @return Nonzero if sin is NaN, zero otherwise.
 */
int sisnan(const float sin)
{
    return isnan(sin);
}
