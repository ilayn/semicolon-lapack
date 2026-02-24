/**
 * @file dsvdct.c
 * @brief DSVDCT counts eigenvalues of a 2*N by 2*N tridiagonal matrix.
 */

#include "verify.h"
#include <math.h>

/**
 * DSVDCT counts the number NUM of eigenvalues of a 2*N by 2*N
 * tridiagonal matrix T which are less than or equal to SHIFT.  T is
 * formed by putting zeros on the diagonal and making the off-diagonals
 * equal to S(1), E(1), S(2), E(2), ... , E(N-1), S(N).  If SHIFT is
 * positive, NUM is equal to N plus the number of singular values of a
 * bidiagonal matrix B less than or equal to SHIFT.  Here B has diagonal
 * entries S(1), ..., S(N) and superdiagonal entries E(1), ... E(N-1).
 * If SHIFT is negative, NUM is equal to the number of singular values
 * of B greater than or equal to -SHIFT.
 *
 * See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
 * Matrix", Report CS41, Computer Science Dept., Stanford University,
 * July 21, 1966
 *
 * @param[in]     n      The dimension of the bidiagonal matrix B.
 * @param[in]     s      Double precision array, dimension (n).
 *                       The diagonal entries of the bidiagonal matrix B.
 * @param[in]     e      Double precision array, dimension (n-1).
 *                       The superdiagonal entries of the bidiagonal matrix B.
 * @param[in]     shift  The shift, used as described under Purpose.
 * @param[out]    num    The number of eigenvalues of T less than or equal to SHIFT.
 */
void dsvdct(const INT n, const f64* s, const f64* e, const f64 shift, INT* num)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    f64 unfl = 2.0 * dlamch("Safe minimum");
    f64 ovfl = ONE / unfl;

    /* Find largest entry */

    f64 mx = fabs(s[0]);
    for (INT i = 0; i < n - 1; i++) {
        f64 as = fabs(s[i + 1]);
        f64 ae = fabs(e[i]);
        if (as > mx) mx = as;
        if (ae > mx) mx = ae;
    }

    if (mx == ZERO) {
        if (shift < ZERO) {
            *num = 0;
        } else {
            *num = 2 * n;
        }
        return;
    }

    /* Compute scale factors as in Kahan's report */

    f64 sun = sqrt(unfl);
    f64 ssun = sqrt(sun);
    f64 sov = sqrt(ovfl);
    f64 tom = ssun * sov;
    f64 m1, m2;
    if (mx <= ONE) {
        m1 = ONE / mx;
        m2 = tom;
    } else {
        m1 = ONE;
        m2 = tom / mx;
    }

    /* Begin counting */

    *num = 0;
    f64 sshift = (shift * m1) * m2;
    f64 u = -sshift;
    if (u <= sun) {
        if (u <= ZERO) {
            (*num)++;
            if (u > -sun)
                u = -sun;
        } else {
            u = sun;
        }
    }
    f64 tmp = (s[0] * m1) * m2;
    u = -tmp * (tmp / u) - sshift;
    if (u <= sun) {
        if (u <= ZERO) {
            (*num)++;
            if (u > -sun)
                u = -sun;
        } else {
            u = sun;
        }
    }
    for (INT i = 0; i < n - 1; i++) {
        tmp = (e[i] * m1) * m2;
        u = -tmp * (tmp / u) - sshift;
        if (u <= sun) {
            if (u <= ZERO) {
                (*num)++;
                if (u > -sun)
                    u = -sun;
            } else {
                u = sun;
            }
        }
        tmp = (s[i + 1] * m1) * m2;
        u = -tmp * (tmp / u) - sshift;
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
