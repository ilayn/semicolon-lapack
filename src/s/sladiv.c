/**
 * @file sladiv.c
 * @brief SLADIV performs complex division in real arithmetic, avoiding
 *        unnecessary overflow.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLADIV performs complex division in real arithmetic
 *
 *                      a + i*b
 *           p + i*q = ---------
 *                      c + i*d
 *
 * The algorithm is due to Michael Baudin and Robert L. Smith
 * and can be found in the paper
 * "A Robust Complex Division in Scilab"
 *
 * @param[in]  a  Double precision scalar.
 * @param[in]  b  Double precision scalar.
 * @param[in]  c  Double precision scalar.
 * @param[in]  d  Double precision scalar.
 *                The scalars a, b, c, and d in the above expression.
 * @param[out] p  Pointer to double, receives real part of result.
 * @param[out] q  Pointer to double, receives imaginary part of result.
 */
void sladiv(const f32 a, const f32 b, const f32 c, const f32 d,
            f32* p, f32* q)
{
    const f32 BS = 2.0f;
    const f32 HALF = 0.5f;
    const f32 TWO = 2.0f;

    f32 aa, bb, cc, dd, ab, cd, s, ov, un, be, eps;

    aa = a;
    bb = b;
    cc = c;
    dd = d;
    ab = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    cd = fabsf(c) > fabsf(d) ? fabsf(c) : fabsf(d);
    s = 1.0f;

    ov = slamch("O");  /* Overflow threshold */
    un = slamch("S");  /* Safe minimum */
    eps = slamch("E"); /* Epsilon */
    be = BS / (eps * eps);

    if (ab >= HALF * ov) {
        aa = HALF * aa;
        bb = HALF * bb;
        s = TWO * s;
    }
    if (cd >= HALF * ov) {
        cc = HALF * cc;
        dd = HALF * dd;
        s = HALF * s;
    }
    if (ab <= un * BS / eps) {
        aa = aa * be;
        bb = bb * be;
        s = s / be;
    }
    if (cd <= un * BS / eps) {
        cc = cc * be;
        dd = dd * be;
        s = s * be;
    }
    if (fabsf(d) <= fabsf(c)) {
        sladiv1(aa, bb, cc, dd, p, q);
    } else {
        sladiv1(bb, aa, dd, cc, p, q);
        *q = -(*q);
    }
    *p = (*p) * s;
    *q = (*q) * s;
}
