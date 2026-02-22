/**
 * @file sladiv1.c
 * @brief SLADIV1 is a helper routine for SLADIV, performing complex division
 *        when |D| <= |C| using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_single.h"

/**
 * SLADIV1 is a helper routine for SLADIV, performing complex division
 * when |D| <= |C| using the Baudin-Smith algorithm.
 *
 * This function computes the real and imaginary parts of (A + i*B)/(C + i*D)
 * when |D| <= |C|. It is part of the robust complex division algorithm
 * described in "A Robust Complex Division in Scilab" by Baudin and Smith.
 *
 * @param[in]  a  Single precision scalar, real part of numerator.
 * @param[in]  b  Single precision scalar, imaginary part of numerator.
 * @param[in]  c  Single precision scalar, real part of denominator.
 * @param[in]  d  Single precision scalar, imaginary part of denominator.
 * @param[out] p  Pointer to single, receives real part of result.
 * @param[out] q  Pointer to single, receives imaginary part of result.
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
