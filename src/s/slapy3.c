/**
 * @file slapy3.c
 * @brief SLAPY3 returns sqrt(x**2+y**2+z**2).
 */

#include "internal_build_defs.h"
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
f32 slapy3(const f32 x, const f32 y, const f32 z)
{
    const f32 ZERO = 0.0f;

    f32 w, xabs, yabs, zabs, hugeval;

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
