/**
 * @file ilaver.c
 * @brief ILAVER returns the LAPACK version number.
 */

 #include "semicolon_lapack_auxiliary.h"


/**
 * ILAVER returns the version number of LAPACK.
 *
 * @param[out] major  Major version number.
 * @param[out] minor  Minor version number.
 * @param[out] patch  Patch level.
*/
void ilaver(INT* major, INT* minor, INT* patch)
{
    *major = 3;
    *minor = 12;
    *patch = 1;

    return;
}
