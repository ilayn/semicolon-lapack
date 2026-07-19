/**
 * @file slapy3.c
 * @brief SLAPY3 returns sqrt(x**2+y**2+z**2).
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAPY3 returns `sqrt(x**2 + y**2 + z**2)`, taking care not to cause
 * unnecessary overflow and unnecessary underflow.
 *
 * @param[in] x Single precision scalar. Specifies the value x.
 * @param[in] y Single precision scalar. Specifies the value y.
 * @param[in] z Single precision scalar. Specifies the value z.
 *
 * @return `sqrt(x**2 + y**2 + z**2)` computed safely.
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
