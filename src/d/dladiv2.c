/**
 * @file dladiv2.c
 * @brief DLADIV2 is a helper routine for DLADIV1, performing part of complex
 *        division using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_double.h"

/**
 * DLADIV2 is a helper routine for DLADIV1, performing part of complex
 * division using the Baudin-Smith algorithm.
 *
 * This function computes `(a + b*r) * t` or `(a + d*(b/c)) * t` depending
 * on the value of `r`.
 *
 * @param[in] a Double precision scalar.
 * @param[in] b Double precision scalar.
 * @param[in] c Double precision scalar.
 * @param[in] d Double precision scalar.
 * @param[in] r Double precision scalar, precomputed as `d/c`.
 * @param[in] t Double precision scalar, precomputed as `1/(c + d*r)`.
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
f64 dladiv2(const f64 a, const f64 b, const f64 c,
               const f64 d, const f64 r, const f64 t)
{
    f64 br;

    if (r != 0.0) {
        br = b * r;
        if (br != 0.0) {
            return (a + br) * t;
        } else {
            return a * t + (b * t) * r;
        }
    } else {
        return (a + d * (b / c)) * t;
    }
}
