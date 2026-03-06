/**
 * @file clctes.c
 * @brief CLCTES returns .TRUE. if the eigenvalue Z/D has negative real part.
 */

#include <complex.h>
#include <math.h>
#include "verify.h"

INT clctes(const c64* z, const c64* d)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    if (crealf(*d) == ZERO && cimagf(*d) == ZERO) {
        return (crealf(*z) < ZERO);
    } else {
        if (crealf(*z) == ZERO || crealf(*d) == ZERO) {
            return (copysignf(ONE, cimagf(*z)) !=
                    copysignf(ONE, cimagf(*d)));
        } else if (cimagf(*z) == ZERO || cimagf(*d) == ZERO) {
            return (copysignf(ONE, crealf(*z)) !=
                    copysignf(ONE, crealf(*d)));
        } else {
            f32 zmax = fmaxf(fabsf(crealf(*z)), fabsf(cimagf(*z)));
            return ((crealf(*z) / zmax) * crealf(*d) +
                    (cimagf(*z) / zmax) * cimagf(*d) < ZERO);
        }
    }
}
