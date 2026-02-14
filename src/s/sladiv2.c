/**
 * @file sladiv2.c
 * @brief SLADIV2 is a helper routine for SLADIV1, performing part of complex
 *        division using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_single.h"

/**
 * SLADIV2 is a helper routine for SLADIV1, performing part of complex
 * division using the Baudin-Smith algorithm.
 *
 * This function computes (A + B*R) * T or (A + D*(B/C)) * T depending
 * on the value of R. It is part of the robust complex division algorithm
 * described in "A Robust Complex Division in Scilab" by Baudin and Smith.
 *
 * @param[in] a  Double precision scalar.
 * @param[in] b  Double precision scalar.
 * @param[in] c  Double precision scalar.
 * @param[in] d  Double precision scalar.
 * @param[in] r  Double precision scalar, precomputed as D/C.
 * @param[in] t  Double precision scalar, precomputed as 1/(C + D*R).
 *
 * @return The computed value for use in complex division.
 */
f32 sladiv2(const f32 a, const f32 b, const f32 c,
               const f32 d, const f32 r, const f32 t)
{
    f32 br;

    if (r != 0.0f) {
        br = b * r;
        if (br != 0.0f) {
            return (a + br) * t;
        } else {
            return a * t + (b * t) * r;
        }
    } else {
        return (a + d * (b / c)) * t;
    }
}
