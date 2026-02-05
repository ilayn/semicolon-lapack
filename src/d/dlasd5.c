/**
 * @file dlasd5.c
 * @brief DLASD5 computes the square root of the i-th eigenvalue of a positive
 *        symmetric rank-one modification of a 2-by-2 diagonal matrix.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

static const double ZERO = 0.0;
static const double ONE = 1.0;
static const double TWO = 2.0;
static const double THREE = 3.0;
static const double FOUR = 4.0;

/**
 * DLASD5 computes the square root of the I-th eigenvalue of a positive
 * symmetric rank-one modification of a 2-by-2 diagonal matrix:
 *
 *     diag(D) * diag(D) + RHO * Z * Z^T
 *
 * The diagonal entries in D are assumed to satisfy 0 <= D[0] < D[1].
 * We also assume RHO > 0 and that the Euclidean norm of Z is one.
 *
 * @param[in]     i       The index of the eigenvalue to be computed. i = 1 or i = 2.
 * @param[in]     D       Array of dimension 2. The original eigenvalues. 0 <= D[0] < D[1].
 * @param[in]     Z       Array of dimension 2. The components of the updating vector.
 * @param[out]    delta   Array of dimension 2. Contains (D[j] - sigma_i) in its j-th component.
 * @param[in]     rho     The scalar in the symmetric updating formula.
 * @param[out]    dsigma  The computed sigma_i, the i-th updated eigenvalue.
 * @param[out]    work    Array of dimension 2. Contains (D[j] + sigma_i) in its j-th component.
 */
void dlasd5(const int i, const double* const restrict D, const double* const restrict Z,
            double* const restrict delta, const double rho, double* dsigma,
            double* const restrict work)
{
    double b, c, del, delsq, tau, w;

    del = D[1] - D[0];
    delsq = del * (D[1] + D[0]);

    if (i == 1) {
        w = ONE + FOUR * rho * (Z[1] * Z[1] / (D[0] + THREE * D[1]) -
                                Z[0] * Z[0] / (THREE * D[0] + D[1])) / del;

        if (w > ZERO) {
            b = delsq + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[0] * Z[0] * delsq;

            /* b > 0, always
             * The following tau is dsigma * dsigma - D[0] * D[0] */
            tau = TWO * c / (b + sqrt(fabs(b * b - FOUR * c)));

            /* The following tau is dsigma - D[0] */
            tau = tau / (D[0] + sqrt(D[0] * D[0] + tau));
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
                tau = -TWO * c / (b + sqrt(b * b + FOUR * c));
            } else {
                tau = (b - sqrt(b * b + FOUR * c)) / TWO;
            }

            /* The following tau is dsigma - D[1] */
            tau = tau / (D[1] + sqrt(fabs(D[1] * D[1] + tau)));
            *dsigma = D[1] + tau;
            delta[0] = -(del + tau);
            delta[1] = -tau;
            work[0] = D[0] + tau + D[1];
            work[1] = TWO * D[1] + tau;
        }
    } else {
        /* Now i == 2 */
        b = -delsq + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
        c = rho * Z[1] * Z[1] * delsq;

        /* The following tau is dsigma * dsigma - D[1] * D[1] */
        if (b > ZERO) {
            tau = (b + sqrt(b * b + FOUR * c)) / TWO;
        } else {
            tau = TWO * c / (-b + sqrt(b * b + FOUR * c));
        }

        /* The following tau is dsigma - D[1] */
        tau = tau / (D[1] + sqrt(D[1] * D[1] + tau));
        *dsigma = D[1] + tau;
        delta[0] = -(del + tau);
        delta[1] = -tau;
        work[0] = D[0] + tau + D[1];
        work[1] = TWO * D[1] + tau;
    }
}
