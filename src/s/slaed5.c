/**
 * @file slaed5.c
 * @brief SLAED5 solves the 2-by-2 secular equation.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAED5 computes the I-th eigenvalue of a symmetric rank-one
 * modification of a 2-by-2 diagonal matrix
 *
 *    diag( D )  +  RHO * Z * transpose(Z) .
 *
 * The diagonal elements in the array D are assumed to satisfy
 *    D(0) < D(1).
 *
 * We also assume RHO > 0 and that the Euclidean norm of the vector
 * Z is one.
 *
 * @param[in]     i      The index of the eigenvalue to be computed. i = 0 or i = 1.
 * @param[in]     D      Double precision array, dimension (2).
 *                       The original eigenvalues. We assume D[0] < D[1].
 * @param[in]     Z      Double precision array, dimension (2).
 *                       The components of the updating vector.
 * @param[out]    delta  Double precision array, dimension (2).
 *                       The vector DELTA contains the information necessary
 *                       to construct the eigenvectors.
 * @param[in]     rho    The scalar in the symmetric updating formula.
 * @param[out]    dlam   The computed lambda_I, the I-th updated eigenvalue.
 */
void slaed5(const INT i, const f32* restrict D,
            const f32* restrict Z, f32* restrict delta,
            const f32 rho, f32* dlam)
{
    f32 b, c, del, tau, temp, w;

    del = D[1] - D[0];

    if (i == 0) {
        w = 1.0f + 2.0f * rho * (Z[1] * Z[1] - Z[0] * Z[0]) / del;
        if (w > 0.0f) {
            b = del + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[0] * Z[0] * del;

            /* B > 0, always */
            tau = 2.0f * c / (b + sqrtf(fabsf(b * b - 4.0f * c)));
            *dlam = D[0] + tau;
            delta[0] = -Z[0] / tau;
            delta[1] = Z[1] / (del - tau);
        } else {
            b = -del + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[1] * Z[1] * del;
            if (b > 0.0f) {
                tau = -2.0f * c / (b + sqrtf(b * b + 4.0f * c));
            } else {
                tau = (b - sqrtf(b * b + 4.0f * c)) / 2.0f;
            }
            *dlam = D[1] + tau;
            delta[0] = -Z[0] / (del + tau);
            delta[1] = -Z[1] / tau;
        }
        temp = sqrtf(delta[0] * delta[0] + delta[1] * delta[1]);
        delta[0] = delta[0] / temp;
        delta[1] = delta[1] / temp;
    } else {
        /* Now i == 1 */
        b = -del + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
        c = rho * Z[1] * Z[1] * del;
        if (b > 0.0f) {
            tau = (b + sqrtf(b * b + 4.0f * c)) / 2.0f;
        } else {
            tau = 2.0f * c / (-b + sqrtf(b * b + 4.0f * c));
        }
        *dlam = D[1] + tau;
        delta[0] = -Z[0] / (del + tau);
        delta[1] = -Z[1] / tau;
        temp = sqrtf(delta[0] * delta[0] + delta[1] * delta[1]);
        delta[0] = delta[0] / temp;
        delta[1] = delta[1] / temp;
    }
}
