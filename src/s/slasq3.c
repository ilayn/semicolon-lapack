/** @file slasq3.c
 * @brief SLASQ3 checks for deflation, computes shift and calls dqds.
 */

#include <math.h>
#include "semicolon_lapack_single.h"


/**
 * SLASQ3 checks for deflation, computes a shift (TAU) and calls dqds.
 * In case of failure it changes shifts, and tries again until output
 * is positive.
 *
 * @param[in]     i0     First index (0-based).
 * @param[in,out] n0     Last index (0-based).
 * @param[in,out] Z      Double precision array, dimension (4*N).
 *                        Z holds the qd array.
 * @param[in,out] pp     PP=0 for ping, PP=1 for pong.
 *                        PP=2 indicates that flipping was applied to the Z array
 *                        and that the initial tests for deflation should not be
 *                        performed.
 * @param[out]    dmin   Minimum value of d.
 * @param[out]    sigma  Sum of shifts used in current segment.
 * @param[in,out] desig  Lower order part of SIGMA.
 * @param[in]     qmax   Maximum value of q.
 * @param[in,out] nfail  Increment NFAIL by 1 each time the shift was too big.
 * @param[in,out] iter   Increment ITER by 1 for each iteration.
 * @param[in,out] ndiv   Increment NDIV by 1 for each division.
 * @param[in]     ieee   Flag for IEEE or non IEEE arithmetic (passed to SLASQ5).
 * @param[in,out] ttype  Shift type.
 * @param[in,out] dmin1  Minimum value of d, excluding D(N0).
 * @param[in,out] dmin2  Minimum value of d, excluding D(N0) and D(N0-1).
 * @param[in,out] dn     d(N0).
 * @param[in,out] dn1    d(N0-1).
 * @param[in,out] dn2    d(N0-2).
 * @param[in,out] g      G is passed as an argument in order to save its
 *                        value between calls to SLASQ3.
 * @param[in,out] tau    This is the shift.
 */
void slasq3(const int i0, int* n0, f32* restrict Z,
            int* pp, f32* dmin, f32* sigma, f32* desig,
            int* nfail, int* iter, int* ndiv,
            const int ieee, int* ttype, f32* dmin1, f32* dmin2,
            f32* dn, f32* dn1, f32* dn2, f32* g, f32* tau)
{
    const f32 CBIAS = 1.50f;
    const f32 ZERO = 0.0f;
    const f32 QURTR = 0.250f;
    const f32 HALF = 0.50f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 HUNDRD = 100.0f;

    int ipn4, j4, n0in, nn;
    f32 eps, s, t, temp, tol, tol2;

    n0in = *n0;
    eps = slamch("P");
    tol = eps * HUNDRD;
    tol2 = tol * tol;

    /* Main deflation loop (Fortran label 10). */
    for (;;) {

        if (*n0 < i0) {
            return;
        }
        if (*n0 == i0) {
            /* Single eigenvalue deflation (Fortran label 20). */
            Z[4 * (*n0)] = Z[4 * (*n0) + *pp] + *sigma;
            *n0 = *n0 - 1;
            continue;
        }

        nn = 4 * (*n0) + *pp + 3;

        if (*n0 == (i0 + 1)) {
            /* Two eigenvalues (Fortran label 40). */
            goto two_eigenvalues;
        }

        /* Check whether E(N0-1) is negligible, 1 eigenvalue. */
        if (Z[nn - 5] > tol2 * (*sigma + Z[nn - 3]) &&
            Z[nn - 2 * (*pp) - 4] > tol2 * Z[nn - 7]) {
            /* Not negligible, check E(N0-2) (Fortran label 30). */
            if (Z[nn - 9] > tol2 * (*sigma) &&
                Z[nn - 2 * (*pp) - 8] > tol2 * Z[nn - 11]) {
                /* Both non-negligible, proceed to shift (Fortran label 50). */
                goto compute_shift;
            }
            /* E(N0-2) is negligible, 2 eigenvalues (Fortran label 40). */
            goto two_eigenvalues;
        }

        /* E(N0-1) is negligible: single eigenvalue deflation (Fortran label 20). */
        Z[4 * (*n0)] = Z[4 * (*n0) + *pp] + *sigma;
        *n0 = *n0 - 1;
        continue;

two_eigenvalues:
        /* Fortran label 40: compute 2 eigenvalues. */
        if (Z[nn - 3] > Z[nn - 7]) {
            s = Z[nn - 3];
            Z[nn - 3] = Z[nn - 7];
            Z[nn - 7] = s;
        }
        t = HALF * ((Z[nn - 7] - Z[nn - 3]) + Z[nn - 5]);
        if (Z[nn - 5] > Z[nn - 3] * tol2 && t != ZERO) {
            s = Z[nn - 3] * (Z[nn - 5] / t);
            if (s <= t) {
                s = Z[nn - 3] * (Z[nn - 5] /
                    (t * (ONE + sqrtf(ONE + s / t))));
            } else {
                s = Z[nn - 3] * (Z[nn - 5] /
                    (t + sqrtf(t) * sqrtf(t + s)));
            }
            t = Z[nn - 7] + (s + Z[nn - 5]);
            Z[nn - 3] = Z[nn - 3] * (Z[nn - 7] / t);
            Z[nn - 7] = t;
        }
        Z[4 * (*n0) - 4] = Z[nn - 7] + *sigma;
        Z[4 * (*n0)] = Z[nn - 3] + *sigma;
        *n0 = *n0 - 2;
        continue;

compute_shift:
        /* Fortran label 50. */
        if (*pp == 2) {
            *pp = 0;
        }

        /* Reverse the qd-array, if warranted. */
        if (*dmin <= ZERO || *n0 < n0in) {
            if (CBIAS * Z[4 * i0 + *pp] < Z[4 * (*n0) + *pp]) {
                ipn4 = 4 * (i0 + *n0);
                for (j4 = 4 * i0; j4 <= 2 * (i0 + *n0 - 1); j4 += 4) {
                    temp = Z[j4];
                    Z[j4] = Z[ipn4 - j4];
                    Z[ipn4 - j4] = temp;

                    temp = Z[j4 + 1];
                    Z[j4 + 1] = Z[ipn4 - j4 + 1];
                    Z[ipn4 - j4 + 1] = temp;

                    temp = Z[j4 + 2];
                    Z[j4 + 2] = Z[ipn4 - j4 - 2];
                    Z[ipn4 - j4 - 2] = temp;

                    temp = Z[j4 + 3];
                    Z[j4 + 3] = Z[ipn4 - j4 - 1];
                    Z[ipn4 - j4 - 1] = temp;
                }
                if (*n0 - i0 <= 4) {
                    Z[4 * (*n0) + *pp + 2] = Z[4 * i0 + *pp + 2];
                    Z[4 * (*n0) - *pp + 3] = Z[4 * i0 - *pp + 3];
                }
                *dmin2 = fminf(*dmin2, Z[4 * (*n0) + *pp + 2]);
                Z[4 * (*n0) + *pp + 2] = fminf(fminf(Z[4 * (*n0) + *pp + 2],
                                                     Z[4 * i0 + *pp + 2]),
                                                Z[4 * i0 + *pp + 6]);
                Z[4 * (*n0) - *pp + 3] = fminf(fminf(Z[4 * (*n0) - *pp + 3],
                                                     Z[4 * i0 - *pp + 3]),
                                                Z[4 * i0 - *pp + 7]);
                *dmin = -ZERO;
            }
        }

        /* Choose a shift. */
        slasq4(i0, *n0, Z, *pp, n0in, *dmin, *dmin1, *dmin2, *dn, *dn1,
                *dn2, tau, ttype, g);

        /* Call dqds until DMIN > 0 (Fortran label 70). */
        for (;;) {

            slasq5(i0, *n0, Z, *pp, *tau, *sigma, dmin, dmin1, dmin2,
                    dn, dn1, dn2, ieee, eps);

            *ndiv = *ndiv + (*n0 - i0 + 2);
            *iter = *iter + 1;

            /* Check status. */
            if (*dmin >= ZERO && *dmin1 >= ZERO) {
                /* Success (Fortran label 90). */
                break;

            } else if (*dmin < ZERO && *dmin1 > ZERO &&
                       Z[4 * (*n0) - *pp - 1] < tol * (*sigma + *dn1) &&
                       fabsf(*dn) < tol * (*sigma)) {
                /* Convergence hidden by negative DN. */
                Z[4 * (*n0) - *pp + 1] = ZERO;
                *dmin = ZERO;
                break;

            } else if (*dmin < ZERO) {
                /* TAU too big. Select new TAU and try again. */
                *nfail = *nfail + 1;
                if (*ttype < -22) {
                    /* Failed twice. Play it safe. */
                    *tau = ZERO;
                } else if (*dmin1 > ZERO) {
                    /* Late failure. Gives excellent shift. */
                    *tau = (*tau + *dmin) * (ONE - TWO * eps);
                    *ttype = *ttype - 11;
                } else {
                    /* Early failure. Divide by 4. */
                    *tau = QURTR * (*tau);
                    *ttype = *ttype - 12;
                }
                continue;  /* Go back to label 70. */

            } else if (sisnan(*dmin)) {
                /* NaN. */
                if (*tau == ZERO) {
                    /* Risk of underflow (Fortran label 80). */
                    goto underflow;
                } else {
                    *tau = ZERO;
                    continue;  /* Go back to label 70. */
                }
            } else {
                /* Possible underflow. Play it safe (Fortran label 80). */
                goto underflow;
            }
        }

        /* Success path (Fortran label 90). */
        goto update_sigma;

underflow:
        /* Fortran label 80: Risk of underflow. */
        slasq6(i0, *n0, Z, *pp, dmin, dmin1, dmin2, dn, dn1, dn2);
        *ndiv = *ndiv + (*n0 - i0 + 2);
        *iter = *iter + 1;
        *tau = ZERO;

update_sigma:
        /* Fortran label 90: update sigma. */
        if (*tau < *sigma) {
            *desig = *desig + *tau;
            t = *sigma + *desig;
            *desig = *desig - (t - *sigma);
        } else {
            t = *sigma + *tau;
            *desig = *sigma - (t - *tau) + *desig;
        }
        *sigma = t;

        return;
    }
}
