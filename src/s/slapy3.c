/**
 * @file slapy3.c
 * @brief SLAPY3 returns sqrt(x**2+y**2+z**2).
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause
 * unnecessary overflow and unnecessary underflow.
 *
 * @param[in] x  First value.
 * @param[in] y  Second value.
 * @param[in] z  Third value.
 * @return       sqrt(x**2 + y**2 + z**2)
 */
float slapy3(const float x, const float y, const float z)
{
    const float ZERO = 0.0f;

    float w, xabs, yabs, zabs, hugeval;

    hugeval = slamch("O");
    xabs = fabsf(x);
    yabs = fabsf(y);
    zabs = fabsf(z);
    w = xabs;
    if (yabs > w) w = yabs;
    if (zabs > w) w = zabs;
    if (w == ZERO || w > hugeval) {
        return xabs + yabs + zabs;
    } else {
        return w * sqrtf((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) +
                        (zabs / w) * (zabs / w));
    }
}
