/**
 * @file slasq5.c
 * @brief SLASQ5 computes one dqds transform in ping-pong form.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLASQ5 computes one dqds transform in ping-pong form, one
 * version for IEEE machines another for non IEEE machines.
 *
 * @param[in]     i0     First index (1-based).
 * @param[in]     n0     Last index (1-based).
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
void slasq5(const int i0, const int n0, float* const restrict Z,
            const int pp, float tau, float sigma,
            float* dmin, float* dmin1, float* dmin2,
            float* dn, float* dnm1, float* dnm2,
            const int ieee, const float eps)
{
    int j4, j4p2;
    float d, emin, temp, dthresh;

    if ((n0 - i0 - 1) <= 0) {
        return;
    }

    dthresh = eps * (sigma + tau);
    if (tau < dthresh * 0.5f) {
        tau = 0.0f;
    }

    if (tau != 0.0f) {
        j4 = 4 * i0 + pp - 3;
        emin = Z[(j4 + 4) - 1];
        d = Z[j4 - 1] - tau;
        *dmin = d;
        *dmin1 = -Z[j4 - 1];

        if (ieee) {
            /* Code for IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 2) - 1] = d + Z[(j4 - 1) - 1];
                    temp = Z[(j4 + 1) - 1] / Z[(j4 - 2) - 1];
                    d = d * temp - tau;
                    *dmin = fminf(*dmin, d);
                    Z[j4 - 1] = Z[(j4 - 1) - 1] * temp;
                    emin = fminf(Z[j4 - 1], emin);
                }
            } else {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 3) - 1] = d + Z[j4 - 1];
                    temp = Z[(j4 + 2) - 1] / Z[(j4 - 3) - 1];
                    d = d * temp - tau;
                    *dmin = fminf(*dmin, d);
                    Z[(j4 - 1) - 1] = Z[j4 - 1] * temp;
                    emin = fminf(Z[(j4 - 1) - 1], emin);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm2 + Z[j4p2 - 1];
            Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
            *dnm1 = Z[(j4p2 + 2) - 1] * (*dnm2 / Z[(j4 - 2) - 1]) - tau;
            *dmin = fminf(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm1 + Z[j4p2 - 1];
            Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
            *dn = Z[(j4p2 + 2) - 1] * (*dnm1 / Z[(j4 - 2) - 1]) - tau;
            *dmin = fminf(*dmin, *dn);

        } else {
            /* Code for non IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 2) - 1] = d + Z[(j4 - 1) - 1];
                    if (d < 0.0f) {
                        return;
                    } else {
                        Z[j4 - 1] = Z[(j4 + 1) - 1] * (Z[(j4 - 1) - 1] / Z[(j4 - 2) - 1]);
                        d = Z[(j4 + 1) - 1] * (d / Z[(j4 - 2) - 1]) - tau;
                    }
                    *dmin = fminf(*dmin, d);
                    emin = fminf(emin, Z[j4 - 1]);
                }
            } else {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 3) - 1] = d + Z[j4 - 1];
                    if (d < 0.0f) {
                        return;
                    } else {
                        Z[(j4 - 1) - 1] = Z[(j4 + 2) - 1] * (Z[j4 - 1] / Z[(j4 - 3) - 1]);
                        d = Z[(j4 + 2) - 1] * (d / Z[(j4 - 3) - 1]) - tau;
                    }
                    *dmin = fminf(*dmin, d);
                    emin = fminf(emin, Z[(j4 - 1) - 1]);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm2 + Z[j4p2 - 1];
            if (*dnm2 < 0.0f) {
                return;
            } else {
                Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
                *dnm1 = Z[(j4p2 + 2) - 1] * (*dnm2 / Z[(j4 - 2) - 1]) - tau;
            }
            *dmin = fminf(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm1 + Z[j4p2 - 1];
            if (*dnm1 < 0.0f) {
                return;
            } else {
                Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
                *dn = Z[(j4p2 + 2) - 1] * (*dnm1 / Z[(j4 - 2) - 1]) - tau;
            }
            *dmin = fminf(*dmin, *dn);
        }
    } else {
        /* This is the version that sets d's to zero if they are small enough */
        j4 = 4 * i0 + pp - 3;
        emin = Z[(j4 + 4) - 1];
        d = Z[j4 - 1] - tau;
        *dmin = d;
        *dmin1 = -Z[j4 - 1];

        if (ieee) {
            /* Code for IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 2) - 1] = d + Z[(j4 - 1) - 1];
                    temp = Z[(j4 + 1) - 1] / Z[(j4 - 2) - 1];
                    d = d * temp - tau;
                    if (d < dthresh) { d = 0.0f; }
                    *dmin = fminf(*dmin, d);
                    Z[j4 - 1] = Z[(j4 - 1) - 1] * temp;
                    emin = fminf(Z[j4 - 1], emin);
                }
            } else {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 3) - 1] = d + Z[j4 - 1];
                    temp = Z[(j4 + 2) - 1] / Z[(j4 - 3) - 1];
                    d = d * temp - tau;
                    if (d < dthresh) { d = 0.0f; }
                    *dmin = fminf(*dmin, d);
                    Z[(j4 - 1) - 1] = Z[j4 - 1] * temp;
                    emin = fminf(Z[(j4 - 1) - 1], emin);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm2 + Z[j4p2 - 1];
            Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
            *dnm1 = Z[(j4p2 + 2) - 1] * (*dnm2 / Z[(j4 - 2) - 1]) - tau;
            *dmin = fminf(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm1 + Z[j4p2 - 1];
            Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
            *dn = Z[(j4p2 + 2) - 1] * (*dnm1 / Z[(j4 - 2) - 1]) - tau;
            *dmin = fminf(*dmin, *dn);

        } else {
            /* Code for non IEEE arithmetic. */
            if (pp == 0) {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 2) - 1] = d + Z[(j4 - 1) - 1];
                    if (d < 0.0f) {
                        return;
                    } else {
                        Z[j4 - 1] = Z[(j4 + 1) - 1] * (Z[(j4 - 1) - 1] / Z[(j4 - 2) - 1]);
                        d = Z[(j4 + 1) - 1] * (d / Z[(j4 - 2) - 1]) - tau;
                    }
                    if (d < dthresh) { d = 0.0f; }
                    *dmin = fminf(*dmin, d);
                    emin = fminf(emin, Z[j4 - 1]);
                }
            } else {
                for (j4 = 4 * i0; j4 <= 4 * (n0 - 3); j4 += 4) {
                    Z[(j4 - 3) - 1] = d + Z[j4 - 1];
                    if (d < 0.0f) {
                        return;
                    } else {
                        Z[(j4 - 1) - 1] = Z[(j4 + 2) - 1] * (Z[j4 - 1] / Z[(j4 - 3) - 1]);
                        d = Z[(j4 + 2) - 1] * (d / Z[(j4 - 3) - 1]) - tau;
                    }
                    if (d < dthresh) { d = 0.0f; }
                    *dmin = fminf(*dmin, d);
                    emin = fminf(emin, Z[(j4 - 1) - 1]);
                }
            }

            /* Unroll last two steps. */
            *dnm2 = d;
            *dmin2 = *dmin;
            j4 = 4 * (n0 - 2) - pp;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm2 + Z[j4p2 - 1];
            if (*dnm2 < 0.0f) {
                return;
            } else {
                Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
                *dnm1 = Z[(j4p2 + 2) - 1] * (*dnm2 / Z[(j4 - 2) - 1]) - tau;
            }
            *dmin = fminf(*dmin, *dnm1);

            *dmin1 = *dmin;
            j4 = j4 + 4;
            j4p2 = j4 + 2 * pp - 1;
            Z[(j4 - 2) - 1] = *dnm1 + Z[j4p2 - 1];
            if (*dnm1 < 0.0f) {
                return;
            } else {
                Z[j4 - 1] = Z[(j4p2 + 2) - 1] * (Z[j4p2 - 1] / Z[(j4 - 2) - 1]);
                *dn = Z[(j4p2 + 2) - 1] * (*dnm1 / Z[(j4 - 2) - 1]) - tau;
            }
            *dmin = fminf(*dmin, *dn);
        }
    }

    Z[(j4 + 2) - 1] = *dn;
    Z[(4 * n0 - pp) - 1] = emin;
}
