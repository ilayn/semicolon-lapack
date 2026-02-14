/**
 * @file dlaed5.c
 * @brief DLAED5 solves the 2-by-2 secular equation.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAED5 computes the I-th eigenvalue of a symmetric rank-one
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
void dlaed5(const int i, const f64* restrict D,
            const f64* restrict Z, f64* restrict delta,
            const f64 rho, f64* dlam)
{
    f64 b, c, del, tau, temp, w;

    del = D[1] - D[0];

    if (i == 0) {
        w = 1.0 + 2.0 * rho * (Z[1] * Z[1] - Z[0] * Z[0]) / del;
        if (w > 0.0) {
            b = del + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[0] * Z[0] * del;

            /* B > 0, always */
            tau = 2.0 * c / (b + sqrt(fabs(b * b - 4.0 * c)));
            *dlam = D[0] + tau;
            delta[0] = -Z[0] / tau;
            delta[1] = Z[1] / (del - tau);
        } else {
            b = -del + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
            c = rho * Z[1] * Z[1] * del;
            if (b > 0.0) {
                tau = -2.0 * c / (b + sqrt(b * b + 4.0 * c));
            } else {
                tau = (b - sqrt(b * b + 4.0 * c)) / 2.0;
            }
            *dlam = D[1] + tau;
            delta[0] = -Z[0] / (del + tau);
            delta[1] = -Z[1] / tau;
        }
        temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        delta[0] = delta[0] / temp;
        delta[1] = delta[1] / temp;
    } else {
        /* Now i == 1 */
        b = -del + rho * (Z[0] * Z[0] + Z[1] * Z[1]);
        c = rho * Z[1] * Z[1] * del;
        if (b > 0.0) {
            tau = (b + sqrt(b * b + 4.0 * c)) / 2.0;
        } else {
            tau = 2.0 * c / (-b + sqrt(b * b + 4.0 * c));
        }
        *dlam = D[1] + tau;
        delta[0] = -Z[0] / (del + tau);
        delta[1] = -Z[1] / tau;
        temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        delta[0] = delta[0] / temp;
        delta[1] = delta[1] / temp;
    }
}
