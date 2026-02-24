/**
 * @file sstect.c
 * @brief SSTECT counts the number of eigenvalues of a tridiagonal matrix
 *        which are less than or equal to SHIFT.
 */

#include <math.h>
#include "verify.h"

/**
 * SSTECT counts the number NUM of eigenvalues of a tridiagonal
 * matrix T which are less than or equal to SHIFT. T has
 * diagonal entries A(0), ... , A(N-1), and offdiagonal entries
 * B(0), ..., B(N-2).
 * See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
 * Matrix", Report CS41, Computer Science Dept., Stanford
 * University, July 21, 1966
 *
 * @param[in]     n      The dimension of the tridiagonal matrix T.
 * @param[in]     a      The diagonal entries of the tridiagonal matrix T, dimension (n).
 * @param[in]     b      The offdiagonal entries, dimension (n-1).
 * @param[in]     shift  The shift value.
 * @param[out]    num    The number of eigenvalues of T less than or equal to SHIFT.
 */
void sstect(const INT n, const f32* a, const f32* b,
            const f32 shift, INT* num)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 THREE = 3.0f;

    INT i;
    f32 m1, m2, mx, ovfl, sov, sshift, ssun, sun, tmp, tom, u, unfl;

    /* Get machine constants */

    unfl = slamch("S");
    ovfl = slamch("O");

    /* Find largest entry */

    mx = fabsf(a[0]);
    for (i = 0; i < n - 1; i++) {
        mx = fmaxf(mx, fmaxf(fabsf(a[i + 1]), fabsf(b[i])));
    }

    /* Handle easy cases, including zero matrix */

    if (shift >= THREE * mx) {
        *num = n;
        return;
    }
    if (shift < -THREE * mx) {
        *num = 0;
        return;
    }

    /* Compute scale factors as in Kahan's report */

    sun = sqrtf(unfl);
    ssun = sqrtf(sun);
    sov = sqrtf(ovfl);
    tom = ssun * sov;
    if (mx <= ONE) {
        m1 = ONE / mx;
        m2 = tom;
    } else {
        m1 = ONE;
        m2 = tom / mx;
    }

    /* Begin counting */

    *num = 0;
    sshift = (shift * m1) * m2;
    u = (a[0] * m1) * m2 - sshift;
    if (u <= sun) {
        if (u <= ZERO) {
            (*num)++;
            if (u > -sun)
                u = -sun;
        } else {
            u = sun;
        }
    }
    for (i = 1; i < n; i++) {
        tmp = (b[i - 1] * m1) * m2;
        u = ((a[i] * m1) * m2 - tmp * (tmp / u)) - sshift;
        if (u <= sun) {
            if (u <= ZERO) {
                (*num)++;
                if (u > -sun)
                    u = -sun;
            } else {
                u = sun;
            }
        }
    }
}
