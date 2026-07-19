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
 * This function computes `(a + b*r) * t` or `(a + d*(b/c)) * t` depending
 * on the value of `r`.
 *
 * @param[in] a Single precision scalar.
 * @param[in] b Single precision scalar.
 * @param[in] c Single precision scalar.
 * @param[in] d Single precision scalar.
 * @param[in] r Single precision scalar, precomputed as `d/c`.
 * @param[in] t Single precision scalar, precomputed as `1/(c + d*r)`.
 *
 * @return The computed value for use in complex division.
 *
 * @par Further Details:
 * @rst
 * The algorithm is due to Michael Baudin and Robert L. Smith
 * and can be found in the paper "A Robust Complex Division in Scilab".
 * https://doi.org/10.48550/arXiv.1210.4539
 * @endrst
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
