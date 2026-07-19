/**
 * @file slapy2.c
 * @brief SLAPY2 returns sqrt(x**2 + y**2), taking care not to cause
 *        unnecessary overflow or underflow.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_single.h"

/**
 * SLAPY2 returns `sqrt(x**2 + y**2)`, taking care not to cause unnecessary
 * overflow and unnecessary underflow.
 *
 * @param[in] x Single precision scalar. Specifies the value x.
 * @param[in] y Single precision scalar. Specifies the value y.
 *
 * @return `sqrt(x**2 + y**2)` computed safely.
 */
f32 slapy2(const f32 x, const f32 y)
{
    f32 xabs, yabs, w, z;

    /* Handle NaN propagation */
    if (isnan(x)) return x;
    if (isnan(y)) return y;

    xabs = fabsf(x);
    yabs = fabsf(y);
    w = xabs > yabs ? xabs : yabs;
    z = xabs < yabs ? xabs : yabs;

    if (z == 0.0f || w > FLT_MAX) {
        return w;
    } else {
        f32 t = z / w;
        return w * sqrtf(1.0f + t * t);
    }
}
