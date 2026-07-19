/**
 * @file dladiv1.c
 * @brief DLADIV1 is a helper routine for DLADIV, performing complex division
 *        when |D| <= |C| using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_double.h"

/**
 * DLADIV1 is a helper routine for DLADIV, performing complex division
 * when `|d| <= |c|` using the Baudin-Smith algorithm.
 *
 * This function computes the real and imaginary parts of `(a + i*b)/(c + i*d)`
 * when `|d| <= |c|`.
 *
 * @param[in]  a Double precision scalar, real part of the numerator.
 * @param[in]  b Double precision scalar, imaginary part of the numerator.
 * @param[in]  c Double precision scalar, real part of the denominator.
 * @param[in]  d Double precision scalar, imaginary part of the denominator.
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
void dladiv1(const f64 a, const f64 b, const f64 c, const f64 d,
             f64* p, f64* q)
{
    f64 r, t;

    r = d / c;
    t = 1.0 / (c + d * r);
    *p = dladiv2(a, b, c, d, r, t);
    *q = dladiv2(b, -a, c, d, r, t);
}
