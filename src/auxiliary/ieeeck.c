/**
 * @file ieeeck.c
 * @brief IEEECK verifies IEEE infinity and NaN arithmetic is safe.
 */

#include "semicolon_lapack_auxiliary.h"

/**
 * IEEECK is called from the ILAENV to verify that Infinity and
 * possibly NaN arithmetic is safe (i.e. will not trap).
 *
 * @param[in] ispec
 *          Specifies whether to test just for infinity arithmetic
 *          or whether to test for infinity and NaN arithmetic.
 *          = 0: Verify infinity arithmetic only.
 *          = 1: Verify infinity and NaN arithmetic.
 *
 * @param[in] zero
 *          Must contain the value 0.0
 *          This is passed to prevent the compiler from optimizing
 *          away this code.
 *
 * @param[in] one
 *          Must contain the value 1.0
 *          This is passed to prevent the compiler from optimizing
 *          away this code.
 *
 * @return
 *          = 0: Arithmetic failed to produce the correct answers
 *          = 1: Arithmetic produced the correct answers
 */
int ieeeck(const int ispec, const float zero, const float one)
{
    float nan1, nan2, nan3, nan4, nan5, nan6;
    float neginf, negzro, newzro, posinf;

    posinf = one / zero;
    if (posinf <= one) {
        return 0;
    }

    neginf = -one / zero;
    if (neginf >= zero) {
        return 0;
    }

    negzro = one / (neginf + one);
    if (negzro != zero) {
        return 0;
    }

    neginf = one / negzro;
    if (neginf >= zero) {
        return 0;
    }

    newzro = negzro + zero;
    if (newzro != zero) {
        return 0;
    }

    posinf = one / newzro;
    if (posinf <= one) {
        return 0;
    }

    neginf = neginf * posinf;
    if (neginf >= zero) {
        return 0;
    }

    posinf = posinf * posinf;
    if (posinf <= one) {
        return 0;
    }

    if (ispec == 0) {
        return 1;
    }

    nan1 = posinf + neginf;
    nan2 = posinf / neginf;
    nan3 = posinf / posinf;
    nan4 = posinf * zero;
    nan5 = neginf * negzro;
    nan6 = nan5 * zero;

    if (nan1 == nan1) {
        return 0;
    }

    if (nan2 == nan2) {
        return 0;
    }

    if (nan3 == nan3) {
        return 0;
    }

    if (nan4 == nan4) {
        return 0;
    }

    if (nan5 == nan5) {
        return 0;
    }

    if (nan6 == nan6) {
        return 0;
    }

    return 1;
}
