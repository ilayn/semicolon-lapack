/**
 * @file slabad.c
 * @brief SLABAD is a no-op kept for compatibility.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"

/**
 * SLABAD is a no-op and kept for compatibility reasons. It used
 * to correct the overflow/underflow behavior of machines that
 * are not IEEE-754 compliant.
 *
 * @param[in,out] small
 *          On entry, the underflow threshold as computed by SLAMCH.
 *          On exit, the unchanged value small.
 *
 * @param[in,out] large
 *          On entry, the overflow threshold as computed by SLAMCH.
 *          On exit, the unchanged value large.
 */
void slabad(f32* small, f32* large)
{
    (void)small;
    (void)large;
}
