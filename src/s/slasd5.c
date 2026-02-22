/**
 * @file slasd5.c
 * @brief SLASD5 computes the square root of the i-th eigenvalue of a positive
 *        symmetric rank-one modification of a 2-by-2 diagonal matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
#include <math.h>

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;
static const f32 TWO = 2.0f;
static const f32 THREE = 3.0f;
static const f32 FOUR = 4.0f;

/**
 * SLASD5 computes the square root of the I-th eigenvalue of a positive
 * symmetric rank-one modification of a 2-by-2 diagonal matrix:
 *
 *     diag(D) * diag(D) + RHO * Z * Z^T
 *
 * The diagonal entries in D are assumed to satisfy 0 <= D[0] < D[1].
 * We also assume RHO > 0 and that the Euclidean norm of Z is one.
 *
 * @param[in]     i       The index of the eigenvalue to be computed. i = 0 or i = 1.
 * @param[in]     D       Array of dimension 2. The original eigenvalues. 0 <= D[0] < D[1].
 * @param[in]     Z       Array of dimension 2. The components of the updating vector.
 * @param[out]    delta   Array of dimension 2. Contains (D[j] - sigma_i) in its j-th component.
 * @param[in]     rho     The scalar in the symmetric updating formula.
 * @param[out]    dsigma  The computed sigma_i, the i-th updated eigenvalue.
 * @param[out]    work    Array of dimension 2. Contains (D[j] + sigma_i) in its j-th component.
 */
void slasd5(const INT i, const f32* restrict D, const f32* restrict Z,
            f32* restrict delta, const f32 rho, f32* dsigma,
            f32* restrict work)
{
    f32 b, c, del, delsq, tau, w;

    del = D[1] - D[0];
    delsq = del * (D[1] + D[0]);

    if (i == 0) {
        w = ONE + FOUR * rho * (Z[1] * Z[1] / (D[0] + THREE * D[1]) -
                                Z[0] * Z[0] / (THREE * D[0] + D[1])) / del;

        if (w > ZERO) {
            b = delsq + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[0] * Z[0] * delsq;

            /* b > 0, always
             * The following tau is dsigma * dsigma - D[0] * D[0] */
            tau = TWO * c / (b + sqrtf(fabsf(b * b - FOUR * c)));

            /* The following tau is dsigma - D[0] */
            tau = tau / (D[0] + sqrtf(D[0] * D[0] + tau));
            *dsigma = D[0] + tau;
            delta[0] = -tau;
            delta[1] = del - tau;
            work[0] = TWO * D[0] + tau;
            work[1] = (D[0] + tau) + D[1];
        } else {
            b = -delsq + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[1] * Z[1] * delsq;

            /* The following tau is dsigma * dsigma - D[1] * D[1] */
            if (b > ZERO) {
                tau = -TWO * c / (b + sqrtf(b * b + FOUR * c));
            } else {
                tau = (b - sqrtf(b * b + FOUR * c)) / TWO;
            }

            /* The following tau is dsigma - D[1] */
            tau = tau / (D[1] + sqrtf(fabsf(D[1] * D[1] + tau)));
            *dsigma = D[1] + tau;
            delta[0] = -(del + tau);
            delta[1] = -tau;
            work[0] = D[0] + tau + D[1];
            work[1] = TWO * D[1] + tau;
        }
    } else {
        /* Now i == 1 */
        b = -delsq + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
        c = rho * Z[1] * Z[1] * delsq;

        /* The following tau is dsigma * dsigma - D[1] * D[1] */
        if (b > ZERO) {
            tau = (b + sqrtf(b * b + FOUR * c)) / TWO;
        } else {
            tau = TWO * c / (-b + sqrtf(b * b + FOUR * c));
        }

        /* The following tau is dsigma - D[1] */
        tau = tau / (D[1] + sqrtf(D[1] * D[1] + tau));
        *dsigma = D[1] + tau;
        delta[0] = -(del + tau);
        delta[1] = -tau;
        work[0] = D[0] + tau + D[1];
        work[1] = TWO * D[1] + tau;
    }
}
