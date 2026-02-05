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
 * @param[in] cmach  Specifies the value to be returned by dlamch:
 *                   = 'E' or 'e':  dlamch := eps (relative machine precision)
 *                   = 'S' or 's':  dlamch := sfmin (safe minimum, 1/sfmin doesn't overflow)
 *                   = 'B' or 'b':  dlamch := base (base of the machine)
 *                   = 'P' or 'p':  dlamch := eps*base
 *                   = 'N' or 'n':  dlamch := t (number of digits in mantissa)
 *                   = 'R' or 'r':  dlamch := rnd (1.0 if rounding, 0.0 otherwise)
 *                   = 'M' or 'm':  dlamch := emin (minimum exponent)
 *                   = 'U' or 'u':  dlamch := rmin (underflow threshold)
 *                   = 'L' or 'l':  dlamch := emax (largest exponent)
 *                   = 'O' or 'o':  dlamch := rmax (overflow threshold)
 *
 * @return The requested machine parameter.
 */
double dlamch(const char* cmach)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    double eps, sfmin, small, rmach;

    // eps = relative machine precision (assuming rounding)
    // DBL_EPSILON is the difference between 1 and the smallest double > 1
    // For rounding arithmetic, divide by 2
    eps = DBL_EPSILON * 0.5;

    if (cmach[0] == 'E' || cmach[0] == 'e') {
        // Epsilon
        rmach = eps;
    } else if (cmach[0] == 'S' || cmach[0] == 's') {
        // Safe minimum
        sfmin = DBL_MIN;
        small = ONE / DBL_MAX;
        if (small >= sfmin) {
            // Use SMALL plus a bit, to avoid the possibility of rounding
            // causing overflow when computing 1/sfmin.
            sfmin = small * (ONE + eps);
        }
        rmach = sfmin;
    } else if (cmach[0] == 'B' || cmach[0] == 'b') {
        // Base
        rmach = (double)FLT_RADIX;
    } else if (cmach[0] == 'P' || cmach[0] == 'p') {
        // Precision = eps * base
        rmach = eps * (double)FLT_RADIX;
    } else if (cmach[0] == 'N' || cmach[0] == 'n') {
        // Number of (base) digits in the mantissa
        rmach = (double)DBL_MANT_DIG;
    } else if (cmach[0] == 'R' || cmach[0] == 'r') {
        // Rounding mode (1.0 for rounding)
        rmach = ONE;
    } else if (cmach[0] == 'M' || cmach[0] == 'm') {
        // Minimum exponent before underflow
        rmach = (double)DBL_MIN_EXP;
    } else if (cmach[0] == 'U' || cmach[0] == 'u') {
        // Underflow threshold
        rmach = DBL_MIN;
    } else if (cmach[0] == 'L' || cmach[0] == 'l') {
        // Largest exponent before overflow
        rmach = (double)DBL_MAX_EXP;
    } else if (cmach[0] == 'O' || cmach[0] == 'o') {
        // Overflow threshold
        rmach = DBL_MAX;
    } else {
        rmach = ZERO;
    }

    return rmach;
}
