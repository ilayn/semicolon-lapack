/**
 * @file ieeeck.c
 * @brief IEEECK verifies IEEE infinity and NaN arithmetic is safe.
 */

#include <float.h>
#include <math.h>
#include "semicolon_lapack_auxiliary.h"

#if FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128
#define IEEE_SUPPORT_INF 1
#else
#define IEEE_SUPPORT_INF 0
#endif

#if IEEE_SUPPORT_INF && defined(NAN)
#define IEEE_SUPPORT_NAN 1
#else
#define IEEE_SUPPORT_NAN 0
#endif

/**
 * IEEECK is called from the ILAENV to verify that Infinity and
 * possibly NaN arithmetic is safe (i.e. will not trap).
 *
 * @param[in] ispec Specifies whether to test just for infinity arithmetic
 *                  or whether to test for infinity and NaN arithmetic.
 *                  `ispec=0`: Verify infinity arithmetic only.
 *                  `ispec=1`: Verify infinity and NaN arithmetic.
 * @param[in] zero  Must contain the value 0.0. This is passed to prevent the
 *                  compiler from optimizing away this code.
 * @param[in] one   Must contain the value 1.0. This is passed to prevent the
 *                  compiler from optimizing away this code.
 *
 * @return `0`: Arithmetic failed to produce the correct answers.
 *         `1`: Arithmetic produced the correct answers.
 */
INT ieeeck(const INT ispec, const float zero, const float one)
{
    (void)zero;
    (void)one;

    if (!IEEE_SUPPORT_INF) {
        return 0;
    }

    if (ispec == 0) {
        return 1;
    }

    if (!IEEE_SUPPORT_NAN) {
        return 0;
    }

    return 1;
}
