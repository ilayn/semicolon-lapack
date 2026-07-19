/**
 * @file slamch.c
 * @brief SLAMCH determines single precision machine parameters.
 */

#include <float.h>
#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAMCH determines single precision machine parameters.
 *
 * @param[in] cmach Specifies the value to be returned by slamch:
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
f32 slamch(const char* cmach)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    f32 eps, sfmin, small, rmach;

    // eps = relative machine precision (assuming rounding)
    // FLT_EPSILON is the difference between 1 and the smallest float > 1
    // For rounding arithmetic, divide by 2
    eps = FLT_EPSILON * 0.5f;

    if (cmach[0] == 'E' || cmach[0] == 'e') {
        // Epsilon
        rmach = eps;
    } else if (cmach[0] == 'S' || cmach[0] == 's') {
        // Safe minimum
        sfmin = FLT_MIN;
        small = ONE / FLT_MAX;
        if (small >= sfmin) {
            // Use SMALL plus a bit, to avoid the possibility of rounding
            // causing overflow when computing 1/sfmin.
            sfmin = small * (ONE + eps);
        }
        rmach = sfmin;
    } else if (cmach[0] == 'B' || cmach[0] == 'b') {
        // Base
        rmach = (f32)FLT_RADIX;
    } else if (cmach[0] == 'P' || cmach[0] == 'p') {
        // Precision = eps * base
        rmach = eps * (f32)FLT_RADIX;
    } else if (cmach[0] == 'N' || cmach[0] == 'n') {
        // Number of (base) digits in the mantissa
        rmach = (f32)FLT_MANT_DIG;
    } else if (cmach[0] == 'R' || cmach[0] == 'r') {
        // Rounding mode (1.0 for rounding)
        rmach = ONE;
    } else if (cmach[0] == 'M' || cmach[0] == 'm') {
        // Minimum exponent before underflow
        rmach = (f32)FLT_MIN_EXP;
    } else if (cmach[0] == 'U' || cmach[0] == 'u') {
        // Underflow threshold
        rmach = FLT_MIN;
    } else if (cmach[0] == 'L' || cmach[0] == 'l') {
        // Largest exponent before overflow
        rmach = (f32)FLT_MAX_EXP;
    } else if (cmach[0] == 'O' || cmach[0] == 'o') {
        // Overflow threshold
        rmach = FLT_MAX;
    } else {
        rmach = ZERO;
    }

    return rmach;
}
