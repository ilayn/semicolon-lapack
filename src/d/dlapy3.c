/**
 * @file dlapy3.c
 * @brief DLAPY3 returns sqrt(x**2+y**2+z**2).
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAPY3 returns `sqrt(x**2 + y**2 + z**2)`, taking care not to cause
 * unnecessary overflow and unnecessary underflow.
 *
 * @param[in] x Double precision scalar. Specifies the value x.
 * @param[in] y Double precision scalar. Specifies the value y.
 * @param[in] z Double precision scalar. Specifies the value z.
 *
 * @return `sqrt(x**2 + y**2 + z**2)` computed safely.
 */
f64 dlapy3(const f64 x, const f64 y, const f64 z)
{
    const f64 ZERO = 0.0;

    f64 w, xabs, yabs, zabs, hugeval;

    hugeval = dlamch("O");
    xabs = fabs(x);
    yabs = fabs(y);
    zabs = fabs(z);
    w = xabs;
    if (yabs > w) w = yabs;
    if (zabs > w) w = zabs;
    if (w == ZERO || w > hugeval) {
        return xabs + yabs + zabs;
    } else {
        return w * sqrt((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) +
                        (zabs / w) * (zabs / w));
    }
}
