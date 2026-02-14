/**
 * @file dladiv1.c
 * @brief DLADIV1 is a helper routine for DLADIV, performing complex division
 *        when |D| <= |C| using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_double.h"

/**
 * DLADIV1 is a helper routine for DLADIV, performing complex division
 * when |D| <= |C| using the Baudin-Smith algorithm.
 *
 * This function computes the real and imaginary parts of (A + i*B)/(C + i*D)
 * when |D| <= |C|. It is part of the robust complex division algorithm
 * described in "A Robust Complex Division in Scilab" by Baudin and Smith.
 *
 * @param[in]  a  Double precision scalar, real part of numerator.
 * @param[in]  b  Double precision scalar, imaginary part of numerator.
 * @param[in]  c  Double precision scalar, real part of denominator.
 * @param[in]  d  Double precision scalar, imaginary part of denominator.
 * @param[out] p  Pointer to f64, receives real part of result.
 * @param[out] q  Pointer to f64, receives imaginary part of result.
 */
void dladiv1(const f64 a, const f64 b, const f64 c, const f64 d,
             f64* p, f64* q)
{
    f64 r, t;

    r = d / c;
    t = 1.0 / (c + d * r);
    *p = dladiv2(a, b, c, d, r, t);
    *q = dladiv2(b, -a, c, d, r, t);
}
