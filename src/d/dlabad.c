/**
 * @file dlabad.c
 * @brief DLABAD is a no-op kept for compatibility.
 */

#include "semicolon_lapack_double.h"

/**
 * DLABAD is a no-op and kept for compatibility reasons. It used
 * to correct the overflow/underflow behavior of machines that
 * are not IEEE-754 compliant.
 *
 * @param[in,out] small
 *          On entry, the underflow threshold as computed by DLAMCH.
 *          On exit, the unchanged value small.
 *
 * @param[in,out] large
 *          On entry, the overflow threshold as computed by DLAMCH.
 *          On exit, the unchanged value large.
 */
void dlabad(double* small, double* large)
{
    (void)small;
    (void)large;
}
