/**
 * @file dlamch.c
 * @brief DLAMCH determines double precision machine parameters.
 */

#include <float.h>
#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAMCH determines double precision machine parameters.
 *
 * @param[in] cmach Specifies the value to be returned by dlamch:
 *                  `cmach='E'`: eps (relative machine precision).
 *                  `cmach='S'`: sfmin (safe minimum, such that `1/sfmin` does not
 *                  overflow).
 *                  `cmach='B'`: base (base of the machine).
 *                  `cmach='P'`: eps*base.
 *                  `cmach='N'`: t (number of digits in the mantissa).
 *                  `cmach='R'`: rnd (1.0 when rounding occurs in addition, 0.0
 *                  otherwise).
 *                  `cmach='M'`: emin (minimum exponent before underflow).
 *                  `cmach='U'`: rmin (underflow threshold).
 *                  `cmach='L'`: emax (largest exponent before overflow).
 *                  `cmach='O'`: rmax (overflow threshold).
 *
 * @return The requested machine parameter.
 */
f64 dlamch(const char* cmach)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    f64 eps, sfmin, small, rmach;

    // Assume rounding, not chopping. Always.
    eps = DBL_EPSILON * 0.5;

    if (cmach[0] == 'E' || cmach[0] == 'e') {
        rmach = eps;
    } else if (cmach[0] == 'S' || cmach[0] == 's') {
        sfmin = DBL_MIN;
        small = ONE / DBL_MAX;
        if (small >= sfmin) {
            // Use SMALL plus a bit, to avoid the possibility of rounding
            // causing overflow when computing 1/sfmin.
            sfmin = small * (ONE + eps);
        }
        rmach = sfmin;
    } else if (cmach[0] == 'B' || cmach[0] == 'b') {
        rmach = (f64)FLT_RADIX;
    } else if (cmach[0] == 'P' || cmach[0] == 'p') {
        rmach = eps * (f64)FLT_RADIX;
    } else if (cmach[0] == 'N' || cmach[0] == 'n') {
        rmach = (f64)DBL_MANT_DIG;
    } else if (cmach[0] == 'R' || cmach[0] == 'r') {
        rmach = ONE;
    } else if (cmach[0] == 'M' || cmach[0] == 'm') {
        rmach = (f64)DBL_MIN_EXP;
    } else if (cmach[0] == 'U' || cmach[0] == 'u') {
        rmach = DBL_MIN;
    } else if (cmach[0] == 'L' || cmach[0] == 'l') {
        rmach = (f64)DBL_MAX_EXP;
    } else if (cmach[0] == 'O' || cmach[0] == 'o') {
        rmach = DBL_MAX;
    } else {
        rmach = ZERO;
    }

    return rmach;
}
