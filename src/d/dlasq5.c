/**
 * @file dlasq5.c
 * @brief DLASQ5 computes one dqds transform in ping-pong form.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLASQ5 computes one dqds transform in ping-pong form, one
 * version for IEEE machines another for non IEEE machines.
 *
 * @param[in]     i0     First index (0-based).
 * @param[in]     n0     Last index (0-based).
 * @param[in,out] Z      Double precision array, dimension (4*N).
 *                        Z holds the qd array. EMIN is stored in Z(4*N0)
 *                        to avoid an extra argument.
 * @param[in]     pp     PP=0 for ping, PP=1 for pong.
 * @param[in]     tau    This is the shift.
 * @param[in]     sigma  This is the accumulated shift up to this step.
 * @param[out]    dmin   Minimum value of d.
 * @param[out]    dmin1  Minimum value of d, excluding D(N0).
 * @param[out]    dmin2  Minimum value of d, excluding D(N0) and D(N0-1).
 * @param[out]    dn     d(N0), the last value of d.
 * @param[out]    dnm1   d(N0-1).
 * @param[out]    dnm2   d(N0-2).
 * @param[in]     ieee   Flag for IEEE or non IEEE arithmetic (1=IEEE, 0=non-IEEE).
 * @param[in]     eps    This is the value of epsilon used.
 */
void dlasq5(const INT i0, const INT n0, f64* restrict Z,
            const INT pp, f64 tau, f64 sigma,
            f64* dmin, f64* dmin1, f64* dmin2,
            f64* dn, f64* dnm1, f64* dnm2,
            const INT ieee, const f64 eps)
{
    INT j4, j4p2;
    f64 d, emin, temp, dthresh;

    if ((n0 - i0 - 1) <= 0) {
        return;
    }

    dthresh = eps * (sigma + tau);
    if (tau < dthresh * 0.5) {
        tau = 0.0;
    }

    if (tau != 0.0) {
        j4 = 4 * i0 + pp;
        emin = Z[j4 + 4];
        d = Z[j4] - tau;
        *dmin = d;
        *dmin1 = -Z[j4];

        if (ieee) {
            /* Code for IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 2] = d + Z[j4 - 1];
                    temp = Z[j4 + 1] / Z[j4 - 2];
                    d = d * temp - tau;
                    *dmin = fmin(*dmin, d);
                    Z[j4] = Z[j4 - 1] * temp;
                    emin = fmin(Z[j4], emin);
                }
            } else {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 3] = d + Z[j4];
                    temp = Z[j4 + 2] / Z[j4 - 3];
                    d = d * temp - tau;
                    *dmin = fmin(*dmin, d);
                    Z[j4 - 1] = Z[j4] * temp;
                    emin = fmin(Z[j4 - 1], emin);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * n0 - pp - 5;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm2 + Z[j4p2];
            Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
            *dnm1 = Z[j4p2 + 2] * (*dnm2 / Z[j4 - 2]) - tau;
            *dmin = fmin(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm1 + Z[j4p2];
            Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
            *dn = Z[j4p2 + 2] * (*dnm1 / Z[j4 - 2]) - tau;
            *dmin = fmin(*dmin, *dn);

        } else {
            /* Code for non IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 2] = d + Z[j4 - 1];
                    if (d < 0.0) {
                        return;
                    } else {
                        Z[j4] = Z[j4 + 1] * (Z[j4 - 1] / Z[j4 - 2]);
                        d = Z[j4 + 1] * (d / Z[j4 - 2]) - tau;
                    }
                    *dmin = fmin(*dmin, d);
                    emin = fmin(emin, Z[j4]);
                }
            } else {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 3] = d + Z[j4];
                    if (d < 0.0) {
                        return;
                    } else {
                        Z[j4 - 1] = Z[j4 + 2] * (Z[j4] / Z[j4 - 3]);
                        d = Z[j4 + 2] * (d / Z[j4 - 3]) - tau;
                    }
                    *dmin = fmin(*dmin, d);
                    emin = fmin(emin, Z[j4 - 1]);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * n0 - pp - 5;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm2 + Z[j4p2];
            if (*dnm2 < 0.0) {
                return;
            } else {
                Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
                *dnm1 = Z[j4p2 + 2] * (*dnm2 / Z[j4 - 2]) - tau;
            }
            *dmin = fmin(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm1 + Z[j4p2];
            if (*dnm1 < 0.0) {
                return;
            } else {
                Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
                *dn = Z[j4p2 + 2] * (*dnm1 / Z[j4 - 2]) - tau;
            }
            *dmin = fmin(*dmin, *dn);
        }
    } else {
        /* This is the version that sets d's to zero if they are small enough */
        j4 = 4 * i0 + pp;
        emin = Z[j4 + 4];
        d = Z[j4] - tau;
        *dmin = d;
        *dmin1 = -Z[j4];

        if (ieee) {
            /* Code for IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 2] = d + Z[j4 - 1];
                    temp = Z[j4 + 1] / Z[j4 - 2];
                    d = d * temp - tau;
                    if (d < dthresh) { d = 0.0; }
                    *dmin = fmin(*dmin, d);
                    Z[j4] = Z[j4 - 1] * temp;
                    emin = fmin(Z[j4], emin);
                }
            } else {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 3] = d + Z[j4];
                    temp = Z[j4 + 2] / Z[j4 - 3];
                    d = d * temp - tau;
                    if (d < dthresh) { d = 0.0; }
                    *dmin = fmin(*dmin, d);
                    Z[j4 - 1] = Z[j4] * temp;
                    emin = fmin(Z[j4 - 1], emin);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * n0 - pp - 5;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm2 + Z[j4p2];
            Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
            *dnm1 = Z[j4p2 + 2] * (*dnm2 / Z[j4 - 2]) - tau;
            *dmin = fmin(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm1 + Z[j4p2];
            Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
            *dn = Z[j4p2 + 2] * (*dnm1 / Z[j4 - 2]) - tau;
            *dmin = fmin(*dmin, *dn);

        } else {
            /* Code for non IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 2] = d + Z[j4 - 1];
                    if (d < 0.0) {
                        return;
                    } else {
                        Z[j4] = Z[j4 + 1] * (Z[j4 - 1] / Z[j4 - 2]);
                        d = Z[j4 + 1] * (d / Z[j4 - 2]) - tau;
                    }
                    if (d < dthresh) { d = 0.0; }
                    *dmin = fmin(*dmin, d);
                    emin = fmin(emin, Z[j4]);
                }
            } else {
                for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
                    Z[j4 - 3] = d + Z[j4];
                    if (d < 0.0) {
                        return;
                    } else {
                        Z[j4 - 1] = Z[j4 + 2] * (Z[j4] / Z[j4 - 3]);
                        d = Z[j4 + 2] * (d / Z[j4 - 3]) - tau;
                    }
                    if (d < dthresh) { d = 0.0; }
                    *dmin = fmin(*dmin, d);
                    emin = fmin(emin, Z[j4 - 1]);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * n0 - pp - 5;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm2 + Z[j4p2];
            if (*dnm2 < 0.0) {
                return;
            } else {
                Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
                *dnm1 = Z[j4p2 + 2] * (*dnm2 / Z[j4 - 2]) - tau;
            }
            *dmin = fmin(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[j4 - 2] = *dnm1 + Z[j4p2];
            if (*dnm1 < 0.0) {
                return;
            } else {
                Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
                *dn = Z[j4p2 + 2] * (*dnm1 / Z[j4 - 2]) - tau;
            }
            *dmin = fmin(*dmin, *dn);
        }
    }

    Z[j4 + 2] = *dn;
    Z[4 * n0 - pp + 3] = emin;
}
