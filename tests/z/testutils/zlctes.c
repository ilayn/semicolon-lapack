/**
 * @file zlctes.c
 * @brief ZLCTES returns .TRUE. if the eigenvalue Z/D has negative real part.
 */

#include <complex.h>
#include <math.h>
#include "verify.h"

INT zlctes(const c128* z, const c128* d)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    if (creal(*d) == ZERO && cimag(*d) == ZERO) {
        return (creal(*z) < ZERO);
    } else {
        if (creal(*z) == ZERO || creal(*d) == ZERO) {
            return (copysign(ONE, cimag(*z)) !=
                    copysign(ONE, cimag(*d)));
        } else if (cimag(*z) == ZERO || cimag(*d) == ZERO) {
            return (copysign(ONE, creal(*z)) !=
                    copysign(ONE, creal(*d)));
        } else {
            f64 zmax = fmax(fabs(creal(*z)), fabs(cimag(*z)));
            return ((creal(*z) / zmax) * creal(*d) +
                    (cimag(*z) / zmax) * cimag(*d) < ZERO);
        }
    }
}
