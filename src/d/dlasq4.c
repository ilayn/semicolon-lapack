/**
 * @file dlasq4.c
 * @brief DLASQ4 computes an approximation to the smallest eigenvalue using values of d from the previous transform.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLASQ4 computes an approximation TAU to the smallest eigenvalue
 * using values of d from the previous transform.
 *
 * @param[in]     i0     First index (1-based).
 * @param[in]     n0     Last index (1-based).
 * @param[in]     Z      Double precision array, dimension (4*N0).
 *                        Z holds the qd array.
 * @param[in]     pp     PP=0 for ping, PP=1 for pong.
 * @param[in]     n0in   The value of N0 at start of EIGTEST (1-based).
 * @param[in]     dmin   Minimum value of d.
 * @param[in]     dmin1  Minimum value of d, excluding D(N0).
 * @param[in]     dmin2  Minimum value of d, excluding D(N0) and D(N0-1).
 * @param[in]     dn     d(N0).
 * @param[in]     dn1    d(N0-1).
 * @param[in]     dn2    d(N0-2).
 * @param[out]    tau    This is the shift.
 * @param[out]    ttype  Shift type.
 * @param[in,out] g      G is passed as an argument in order to save its
 *                        value between calls to DLASQ4.
 */
void dlasq4(const int i0, const int n0, const f64* restrict Z,
            const int pp, const int n0in, const f64 dmin,
            const f64 dmin1, const f64 dmin2, const f64 dn,
            const f64 dn1, const f64 dn2, f64* tau,
            int* ttype, f64* g)
{
    /* Constants from the Fortran source: CNST1 = 9/16 = 0.5630 (approx) */
    const f64 CNST1 = 0.5630;
    const f64 CNST2 = 1.010;
    const f64 CNST3 = 1.050;
    const f64 QURTR = 0.250;
    const f64 THIRD = 0.3330;
    const f64 HALF = 0.50;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HUNDRD = 100.0;

    int i4, nn, np;
    f64 a2, b1, b2, gam, gap1, gap2, s = ZERO;

    /*
     * A negative DMIN forces the shift to take that absolute value.
     * TTYPE records the type of shift.
     */
    if (dmin <= ZERO) {
        *tau = -dmin;
        *ttype = -1;
        return;
    }

    nn = 4 * n0 + pp;

    if (n0in == n0) {
        /*
         * No eigenvalues deflated.
         */
        if (dmin == dn || dmin == dn1) {

            b1 = sqrt(Z[(nn - 3) - 1]) * sqrt(Z[(nn - 5) - 1]);
            b2 = sqrt(Z[(nn - 7) - 1]) * sqrt(Z[(nn - 9) - 1]);
            a2 = Z[(nn - 7) - 1] + Z[(nn - 5) - 1];

            /* Cases 2 and 3. */
            if (dmin == dn && dmin1 == dn1) {
                gap2 = dmin2 - a2 - dmin2 * QURTR;
                if (gap2 > ZERO && gap2 > b2) {
                    gap1 = a2 - dn - (b2 / gap2) * b2;
                } else {
                    gap1 = a2 - dn - (b1 + b2);
                }
                if (gap1 > ZERO && gap1 > b1) {
                    s = fmax(dn - (b1 / gap1) * b1, HALF * dmin);
                    *ttype = -2;
                } else {
                    s = ZERO;
                    if (dn > b1) {
                        s = dn - b1;
                    }
                    if (a2 > (b1 + b2)) {
                        s = fmin(s, a2 - (b1 + b2));
                    }
                    s = fmax(s, THIRD * dmin);
                    *ttype = -3;
                }
            } else {
                /* Case 4. */
                *ttype = -4;
                s = QURTR * dmin;
                if (dmin == dn) {
                    gam = dn;
                    a2 = ZERO;
                    if (Z[(nn - 5) - 1] > Z[(nn - 7) - 1]) {
                        return;
                    }
                    b2 = Z[(nn - 5) - 1] / Z[(nn - 7) - 1];
                    np = nn - 9;
                } else {
                    np = nn - 2 * pp;
                    gam = dn1;
                    if (Z[(np - 4) - 1] > Z[(np - 2) - 1]) {
                        return;
                    }
                    a2 = Z[(np - 4) - 1] / Z[(np - 2) - 1];
                    if (Z[(nn - 9) - 1] > Z[(nn - 11) - 1]) {
                        return;
                    }
                    b2 = Z[(nn - 9) - 1] / Z[(nn - 11) - 1];
                    np = nn - 13;
                }

                /* Approximate contribution to norm squared from I < NN-1. */
                a2 = a2 + b2;
                for (i4 = np; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
                    if (b2 == ZERO) {
                        break;
                    }
                    b1 = b2;
                    if (Z[i4 - 1] > Z[(i4 - 2) - 1]) {
                        return;
                    }
                    b2 = b2 * (Z[i4 - 1] / Z[(i4 - 2) - 1]);
                    a2 = a2 + b2;
                    if (HUNDRD * fmax(b2, b1) < a2 || CNST1 < a2) {
                        break;
                    }
                }
                a2 = CNST3 * a2;

                /* Rayleigh quotient residual bound. */
                if (a2 < CNST1) {
                    s = gam * (ONE - sqrt(a2)) / (ONE + a2);
                }
            }
        } else if (dmin == dn2) {
            /* Case 5. */
            *ttype = -5;
            s = QURTR * dmin;

            /* Compute contribution to norm squared from I > NN-2. */
            np = nn - 2 * pp;
            b1 = Z[(np - 2) - 1];
            b2 = Z[(np - 6) - 1];
            gam = dn2;
            if (Z[(np - 8) - 1] > b2 || Z[(np - 4) - 1] > b1) {
                return;
            }
            a2 = (Z[(np - 8) - 1] / b2) * (ONE + Z[(np - 4) - 1] / b1);

            /* Approximate contribution to norm squared from I < NN-2. */
            if (n0 - i0 > 2) {
                b2 = Z[(nn - 13) - 1] / Z[(nn - 15) - 1];
                a2 = a2 + b2;
                for (i4 = nn - 17; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
                    if (b2 == ZERO) {
                        break;
                    }
                    b1 = b2;
                    if (Z[i4 - 1] > Z[(i4 - 2) - 1]) {
                        return;
                    }
                    b2 = b2 * (Z[i4 - 1] / Z[(i4 - 2) - 1]);
                    a2 = a2 + b2;
                    if (HUNDRD * fmax(b2, b1) < a2 || CNST1 < a2) {
                        break;
                    }
                }
                a2 = CNST3 * a2;
            }

            if (a2 < CNST1) {
                s = gam * (ONE - sqrt(a2)) / (ONE + a2);
            }
        } else {
            /* Case 6, no information to guide us. */
            if (*ttype == -6) {
                *g = *g + THIRD * (ONE - *g);
            } else if (*ttype == -18) {
                *g = QURTR * THIRD;
            } else {
                *g = QURTR;
            }
            s = *g * dmin;
            *ttype = -6;
        }

    } else if (n0in == (n0 + 1)) {
        /*
         * One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
         */
        if (dmin1 == dn1 && dmin2 == dn2) {
            /* Cases 7 and 8. */
            *ttype = -7;
            s = THIRD * dmin1;
            if (Z[(nn - 5) - 1] > Z[(nn - 7) - 1]) {
                return;
            }
            b1 = Z[(nn - 5) - 1] / Z[(nn - 7) - 1];
            b2 = b1;
            if (b2 == ZERO) {
                goto label_60;
            }
            for (i4 = 4 * n0 - 9 + pp; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
                a2 = b1;
                if (Z[i4 - 1] > Z[(i4 - 2) - 1]) {
                    return;
                }
                b1 = b1 * (Z[i4 - 1] / Z[(i4 - 2) - 1]);
                b2 = b2 + b1;
                if (HUNDRD * fmax(b1, a2) < b2) {
                    break;
                }
            }
label_60:
            b2 = sqrt(CNST3 * b2);
            a2 = dmin1 / (ONE + b2 * b2);
            gap2 = HALF * dmin2 - a2;
            if (gap2 > ZERO && gap2 > b2 * a2) {
                s = fmax(s, a2 * (ONE - CNST2 * a2 * (b2 / gap2) * b2));
            } else {
                s = fmax(s, a2 * (ONE - CNST2 * b2));
                *ttype = -8;
            }
        } else {
            /* Case 9. */
            s = QURTR * dmin1;
            if (dmin1 == dn1) {
                s = HALF * dmin1;
            }
            *ttype = -9;
        }

    } else if (n0in == (n0 + 2)) {
        /*
         * Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
         *
         * Cases 10 and 11.
         */
        if (dmin2 == dn2 && TWO * Z[(nn - 5) - 1] < Z[(nn - 7) - 1]) {
            *ttype = -10;
            s = THIRD * dmin2;
            if (Z[(nn - 5) - 1] > Z[(nn - 7) - 1]) {
                return;
            }
            b1 = Z[(nn - 5) - 1] / Z[(nn - 7) - 1];
            b2 = b1;
            if (b2 == ZERO) {
                goto label_80;
            }
            for (i4 = 4 * n0 - 9 + pp; i4 >= 4 * i0 - 1 + pp; i4 -= 4) {
                if (Z[i4 - 1] > Z[(i4 - 2) - 1]) {
                    return;
                }
                b1 = b1 * (Z[i4 - 1] / Z[(i4 - 2) - 1]);
                b2 = b2 + b1;
                if (HUNDRD * b1 < b2) {
                    break;
                }
            }
label_80:
            b2 = sqrt(CNST3 * b2);
            a2 = dmin2 / (ONE + b2 * b2);
            gap2 = Z[(nn - 7) - 1] + Z[(nn - 9) - 1]
                   - sqrt(Z[(nn - 11) - 1]) * sqrt(Z[(nn - 9) - 1]) - a2;
            if (gap2 > ZERO && gap2 > b2 * a2) {
                s = fmax(s, a2 * (ONE - CNST2 * a2 * (b2 / gap2) * b2));
            } else {
                s = fmax(s, a2 * (ONE - CNST2 * b2));
            }
        } else {
            s = QURTR * dmin2;
            *ttype = -11;
        }
    } else if (n0in > (n0 + 2)) {
        /*
         * Case 12, more than two eigenvalues deflated. No information.
         */
        s = ZERO;
        *ttype = -12;
    }

    *tau = s;
}
