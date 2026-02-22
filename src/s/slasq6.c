/**
 * @file slasq6.c
 * @brief SLASQ6 computes one dqd transform in ping-pong form.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLASQ6 computes one dqd (shift equal to zero) transform in
 * ping-pong form, with protection against underflow and overflow.
 *
 * @param[in]     i0     First index (0-based).
 * @param[in]     n0     Last index (0-based).
 * @param[in,out] Z      Double precision array, dimension (4*N).
 *                        Z holds the qd array. EMIN is stored in Z(4*N0)
 *                        to avoid an extra argument.
 * @param[in]     pp     PP=0 for ping, PP=1 for pong.
 * @param[out]    dmin   Minimum value of d.
 * @param[out]    dmin1  Minimum value of d, excluding D(N0).
 * @param[out]    dmin2  Minimum value of d, excluding D(N0) and D(N0-1).
 * @param[out]    dn     d(N0), the last value of d.
 * @param[out]    dnm1   d(N0-1).
 * @param[out]    dnm2   d(N0-2).
 */
void slasq6(const INT i0, const INT n0, f32* restrict Z,
            const INT pp, f32* dmin, f32* dmin1, f32* dmin2,
            f32* dn, f32* dnm1, f32* dnm2)
{
    INT j4, j4p2;
    f32 d, emin, safmin, temp;

    if ((n0 - i0 - 1) <= 0) {
        return;
    }

    safmin = slamch("S");
    j4 = 4 * i0 + pp;
    emin = Z[j4 + 4];
    d = Z[j4];
    *dmin = d;

    if (pp == 0) {
        for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
            Z[j4 - 2] = d + Z[j4 - 1];
            if (Z[j4 - 2] == 0.0f) {
                Z[j4] = 0.0f;
                d = Z[j4 + 1];
                *dmin = d;
                emin = 0.0f;
            } else if (safmin * Z[j4 + 1] < Z[j4 - 2] &&
                       safmin * Z[j4 - 2] < Z[j4 + 1]) {
                temp = Z[j4 + 1] / Z[j4 - 2];
                Z[j4] = Z[j4 - 1] * temp;
                d = d * temp;
            } else {
                Z[j4] = Z[j4 + 1] * (Z[j4 - 1] / Z[j4 - 2]);
                d = Z[j4 + 1] * (d / Z[j4 - 2]);
            }
            *dmin = fminf(*dmin, d);
            emin = fminf(emin, Z[j4]);
        }
    } else {
        for (j4 = 4 * i0 + 3; j4 <= 4 * n0 - 9; j4 += 4) {
            Z[j4 - 3] = d + Z[j4];
            if (Z[j4 - 3] == 0.0f) {
                Z[j4 - 1] = 0.0f;
                d = Z[j4 + 2];
                *dmin = d;
                emin = 0.0f;
            } else if (safmin * Z[j4 + 2] < Z[j4 - 3] &&
                       safmin * Z[j4 - 3] < Z[j4 + 2]) {
                temp = Z[j4 + 2] / Z[j4 - 3];
                Z[j4 - 1] = Z[j4] * temp;
                d = d * temp;
            } else {
                Z[j4 - 1] = Z[j4 + 2] * (Z[j4] / Z[j4 - 3]);
                d = Z[j4 + 2] * (d / Z[j4 - 3]);
            }
            *dmin = fminf(*dmin, d);
            emin = fminf(emin, Z[j4 - 1]);
        }
    }

    /* Unroll last two steps. */
    *dnm2 = d;
    *dmin2 = *dmin;
    j4 = 4 * n0 - pp - 5;
    j4p2 = j4 + 2 * pp - 1;
    Z[j4 - 2] = *dnm2 + Z[j4p2];
    if (Z[j4 - 2] == 0.0f) {
        Z[j4] = 0.0f;
        *dnm1 = Z[j4p2 + 2];
        *dmin = *dnm1;
        emin = 0.0f;
    } else if (safmin * Z[j4p2 + 2] < Z[j4 - 2] &&
               safmin * Z[j4 - 2] < Z[j4p2 + 2]) {
        temp = Z[j4p2 + 2] / Z[j4 - 2];
        Z[j4] = Z[j4p2] * temp;
        *dnm1 = *dnm2 * temp;
    } else {
        Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
        *dnm1 = Z[j4p2 + 2] * (*dnm2 / Z[j4 - 2]);
    }
    *dmin = fminf(*dmin, *dnm1);

    *dmin1 = *dmin;
    j4 = j4 + 4;
    j4p2 = j4 + 2 * pp - 1;
    Z[j4 - 2] = *dnm1 + Z[j4p2];
    if (Z[j4 - 2] == 0.0f) {
        Z[j4] = 0.0f;
        *dn = Z[j4p2 + 2];
        *dmin = *dn;
        emin = 0.0f;
    } else if (safmin * Z[j4p2 + 2] < Z[j4 - 2] &&
               safmin * Z[j4 - 2] < Z[j4p2 + 2]) {
        temp = Z[j4p2 + 2] / Z[j4 - 2];
        Z[j4] = Z[j4p2] * temp;
        *dn = *dnm1 * temp;
    } else {
        Z[j4] = Z[j4p2 + 2] * (Z[j4p2] / Z[j4 - 2]);
        *dn = Z[j4p2 + 2] * (*dnm1 / Z[j4 - 2]);
    }
    *dmin = fminf(*dmin, *dn);

    Z[j4 + 2] = *dn;
    Z[4 * n0 - pp + 3] = emin;
}
