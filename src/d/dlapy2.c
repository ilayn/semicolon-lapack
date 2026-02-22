/**
 * @file dlapy2.c
 * @brief DLAPY2 returns sqrt(x**2 + y**2), taking care not to cause
 *        unnecessary overflow or underflow.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DLAPY2 returns sqrt(x**2 + y**2), taking care not to cause unnecessary
 * overflow and unnecessary underflow.
 *
 * @param[in] x  Double precision scalar.
 * @param[in] y  Double precision scalar.
 *             x and y specify the values x and y.
 *
 * @return sqrt(x**2 + y**2) computed safely.
 */
f64 dlapy2(const f64 x, const f64 y)
{
    f64 xabs, yabs, w, z;

    /* Handle NaN propagation */
    if (isnan(x)) return x;
    if (isnan(y)) return y;

    xabs = fabs(x);
    yabs = fabs(y);
    w = xabs > yabs ? xabs : yabs;
    z = xabs < yabs ? xabs : yabs;

    if (z == 0.0 || w > DBL_MAX) {
        return w;
    } else {
        f64 t = z / w;
        return w * sqrt(1.0 + t * t);
    }
}
