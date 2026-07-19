/**
 * @file dladiv.c
 * @brief DLADIV performs complex division in real arithmetic, avoiding
 *        unnecessary overflow.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLADIV performs complex division in real arithmetic
 * @rst
 * .. code-block:: text
 *
 *                  a + i*b
 *       p + i*q = ---------
 *                  c + i*d
 * @endrst
 *
 * @param[in]  a Double precision scalar. The scalar a in the above expression.
 * @param[in]  b Double precision scalar. The scalar b in the above expression.
 * @param[in]  c Double precision scalar. The scalar c in the above expression.
 * @param[in]  d Double precision scalar. The scalar d in the above expression.
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
void dladiv(const f64 a, const f64 b, const f64 c, const f64 d,
            f64* p, f64* q)
{
    const f64 BS = 2.0;
    const f64 HALF = 0.5;
    const f64 TWO = 2.0;

    f64 aa, bb, cc, dd, ab, cd, s, ov, un, be, eps;

    aa = a;
    bb = b;
    cc = c;
    dd = d;
    ab = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    cd = fabs(c) > fabs(d) ? fabs(c) : fabs(d);
    s = 1.0;

    ov = dlamch("O");  /* Overflow threshold */
    un = dlamch("S");  /* Safe minimum */
    eps = dlamch("E"); /* Epsilon */
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
    if (fabs(d) <= fabs(c)) {
        dladiv1(aa, bb, cc, dd, p, q);
    } else {
        dladiv1(bb, aa, dd, cc, p, q);
        *q = -(*q);
    }
    *p = (*p) * s;
    *q = (*q) * s;
}
