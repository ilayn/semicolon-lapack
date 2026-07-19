/**
 * @file sladiv1.c
 * @brief SLADIV1 is a helper routine for SLADIV, performing complex division
 *        when |D| <= |C| using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_single.h"

/**
 * SLADIV1 is a helper routine for SLADIV, performing complex division
 * when `|d| <= |c|` using the Baudin-Smith algorithm.
 *
 * This function computes the real and imaginary parts of `(a + i*b)/(c + i*d)`
 * when `|d| <= |c|`.
 *
 * @param[in]  a Single precision scalar, real part of the numerator.
 * @param[in]  b Single precision scalar, imaginary part of the numerator.
 * @param[in]  c Single precision scalar, real part of the denominator.
 * @param[in]  d Single precision scalar, imaginary part of the denominator.
 * @param[out] p Receives the real part of the result.
 * @param[out] q Receives the imaginary part of the result.
 *
 * @par Further Details:
 * @rst
 * The algorithm is due to Michael Baudin and Robert L. Smith
 * and can be found in the paper "A Robust Complex Division in Scilab".
 * https://doi.org/10.48550/arXiv.1210.4539
 * @endrst
 */
void sladiv1(const f32 a, const f32 b, const f32 c, const f32 d,
             f32* p, f32* q)
{
    f32 r, t;

    r = d / c;
    t = 1.0f / (c + d * r);
    *p = sladiv2(a, b, c, d, r, t);
    *q = sladiv2(b, -a, c, d, r, t);
}
