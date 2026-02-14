/**
 * @file slasq2.c
 * @brief SLASQ2 computes all eigenvalues of a symmetric positive definite tridiagonal matrix.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLASQ2 computes all the eigenvalues of the symmetric positive
 * definite tridiagonal matrix associated with the qd array Z to high
 * relative accuracy are computed to high relative accuracy, in the
 * absence of denormalization, underflow and overflow.
 *
 * To see the relation of Z to the tridiagonal matrix, let L be a
 * unit lower bidiagonal matrix with subdiagonals Z(2,4,6,..) and
 * let U be an upper bidiagonal matrix with 1's above and diagonal
 * Z(1,3,5,..). The tridiagonal is L*U or, if you prefer, the
 * symmetric tridiagonal to which it is similar.
 *
 * @param[in]     n     The number of rows and columns in the matrix. n >= 0.
 * @param[in,out] Z     Double precision array, dimension (4*n).
 *                      On entry Z holds the qd array. On exit, entries 1 to n
 *                      hold the eigenvalues in decreasing order, Z(2*n+1) holds
 *                      the trace, and Z(2*n+2) holds the sum of the eigenvalues.
 *                      If n > 2, then Z(2*n+3) holds the iteration count,
 *                      Z(2*n+4) holds NDIVS/NIN^2, and Z(2*n+5) holds the
 *                      percentage of shifts that failed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if the i-th argument is a scalar and had an illegal
 *                           value, then info = -i, if the i-th argument is an
 *                           array and the j-entry had an illegal value, then
 *                           info = -(i*100+j)
 *                         - > 0: the algorithm failed
 *                         - = 1, a split was marked by a positive value in E
 *                         - = 2, current block of Z not diagonalized after 100*N
 *                           iterations (in inner while loop). On exit Z holds
 *                           a qd array with the same eigenvalues as the given Z.
 *                         - = 3, termination criterion of outer while loop not met
 *                           (program created more than N unreduced blocks)
 */
void slasq2(const int n, f32* restrict Z, int* info)
{
    /* Constants from the Fortran source */
    const f32 CBIAS  = 1.50f;
    const f32 ZERO   = 0.0f;
    const f32 HALF   = 0.5f;
    const f32 ONE    = 1.0f;
    const f32 TWO    = 2.0f;
    const f32 FOUR   = 4.0f;
    const f32 HUNDRD = 100.0f;

    /* Local scalars */
    int ieee;
    int i0, i1, i4, iinfo, ipn4, iter, iwhila, iwhilb,
        k, kmin, n0, n1, nbig, ndiv, nfail, pp, splt, ttype;
    f32 d, dee, deemin, desig, dmin, dmin1, dmin2, dn,
           dn1, dn2, e, emax, emin, eps, g, oldemn, qmax,
           qmin, s, safmin, sigma, t, tau, temp, tol,
           tol2, trace, zmax, tempe, tempq;

    /* Test the input arguments. */
    *info = 0;
    eps = slamch("Precision");
    safmin = slamch("Safe minimum");
    tol = eps * HUNDRD;
    tol2 = tol * tol;

    if (n < 0) {
        *info = -1;
        xerbla("SLASQ2", 1);
        return;
    } else if (n == 0) {
        return;
    } else if (n == 1) {
        /* 1-by-1 case. */
        if (Z[0] < ZERO) {
            *info = -201;
            xerbla("SLASQ2", 2);
        }
        return;
    } else if (n == 2) {
        /* 2-by-2 case. */
        if (Z[0] < ZERO) {
            *info = -201;
            xerbla("SLASQ2", 2);
            return;
        } else if (Z[1] < ZERO) {
            *info = -202;
            xerbla("SLASQ2", 2);
            return;
        } else if (Z[2] < ZERO) {
            *info = -203;
            xerbla("SLASQ2", 2);
            return;
        } else if (Z[2] > Z[0]) {
            d = Z[2];
            Z[2] = Z[0];
            Z[0] = d;
        }
        Z[4] = Z[0] + Z[1] + Z[2];
        if (Z[1] > Z[2] * tol2) {
            t = HALF * ((Z[0] - Z[2]) + Z[1]);
            s = Z[2] * (Z[1] / t);
            if (s <= t) {
                s = Z[2] * (Z[1] / (t * (ONE + sqrtf(ONE + s / t))));
            } else {
                s = Z[2] * (Z[1] / (t + sqrtf(t) * sqrtf(t + s)));
            }
            t = Z[0] + (s + Z[1]);
            Z[2] = Z[2] * (Z[0] / t);
            Z[0] = t;
        }
        Z[1] = Z[2];
        Z[5] = Z[1] + Z[0];
        return;
    }

    /*
     * Check for negative data and compute sums of q's and e's.
     */
    Z[2 * n - 1] = ZERO;
    emin = Z[1];
    qmax = ZERO;
    zmax = ZERO;
    d = ZERO;
    e = ZERO;

    for (k = 1; k <= 2 * (n - 1); k += 2) {
        /* Z(k) -> Z[k-1], Z(k+1) -> Z[k] */
        if (Z[k - 1] < ZERO) {
            *info = -(200 + k);
            xerbla("SLASQ2", 2);
            return;
        } else if (Z[k] < ZERO) {
            *info = -(200 + k + 1);
            xerbla("SLASQ2", 2);
            return;
        }
        d = d + Z[k - 1];
        e = e + Z[k];
        qmax = fmaxf(qmax, Z[k - 1]);
        emin = fminf(emin, Z[k]);
        zmax = fmaxf(fmaxf(qmax, zmax), Z[k]);
    }
    if (Z[2 * n - 2] < ZERO) {
        *info = -(200 + 2 * n - 1);
        xerbla("SLASQ2", 2);
        return;
    }
    d = d + Z[2 * n - 2];
    qmax = fmaxf(qmax, Z[2 * n - 2]);
    zmax = fmaxf(qmax, zmax);

    /* Check for diagonality. */
    if (e == ZERO) {
        for (k = 2; k <= n; k++) {
            Z[k - 1] = Z[2 * k - 2];
        }
        slasrt("D", n, Z, &iinfo);
        Z[2 * n - 2] = d;
        return;
    }

    trace = d + e;

    /* Check for zero data. */
    if (trace == ZERO) {
        Z[2 * n - 2] = ZERO;
        return;
    }

    /*
     * Check whether the machine is IEEE conformable.
     * In this project, we assume IEEE=1 (all modern platforms are IEEE 754).
     * Fortran: IEEE = (ILAENV(10, 'SLASQ2', 'N', 1, 2, 3, 4) == 1)
     */
    ieee = 1;

    /*
     * Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
     */
    for (k = 2 * n; k >= 2; k -= 2) {
        Z[2 * k - 1] = ZERO;
        Z[2 * k - 2] = Z[k - 1];
        Z[2 * k - 3] = ZERO;
        Z[2 * k - 4] = Z[k - 2];
    }

    i0 = 1;
    n0 = n;

    /* Reverse the qd-array, if warranted. */
    if (CBIAS * Z[4 * i0 - 4] < Z[4 * n0 - 4]) {
        ipn4 = 4 * (i0 + n0);
        for (i4 = 4 * i0; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
            temp = Z[i4 - 4];
            Z[i4 - 4] = Z[ipn4 - i4 - 4];
            Z[ipn4 - i4 - 4] = temp;
            temp = Z[i4 - 2];
            Z[i4 - 2] = Z[ipn4 - i4 - 6];
            Z[ipn4 - i4 - 6] = temp;
        }
    }

    /*
     * Initial split checking via dqd and Li's test.
     */
    pp = 0;

    for (k = 1; k <= 2; k++) {

        d = Z[4 * n0 + pp - 4];
        for (i4 = 4 * (n0 - 1) + pp; i4 >= 4 * i0 + pp; i4 -= 4) {
            if (Z[i4 - 2] <= tol2 * d) {
                Z[i4 - 2] = -ZERO;
                d = Z[i4 - 4];
            } else {
                d = Z[i4 - 4] * (d / (d + Z[i4 - 2]));
            }
        }

        /* dqd maps Z to ZZ plus Li's test. */
        emin = Z[4 * i0 + pp];
        d = Z[4 * i0 + pp - 4];
        for (i4 = 4 * i0 + pp; i4 <= 4 * (n0 - 1) + pp; i4 += 4) {
            Z[i4 - 2 * pp - 3] = d + Z[i4 - 2];
            if (Z[i4 - 2] <= tol2 * d) {
                Z[i4 - 2] = -ZERO;
                Z[i4 - 2 * pp - 3] = d;
                Z[i4 - 2 * pp - 1] = ZERO;
                d = Z[i4];
            } else if (safmin * Z[i4] < Z[i4 - 2 * pp - 3] &&
                       safmin * Z[i4 - 2 * pp - 3] < Z[i4]) {
                temp = Z[i4] / Z[i4 - 2 * pp - 3];
                Z[i4 - 2 * pp - 1] = Z[i4 - 2] * temp;
                d = d * temp;
            } else {
                Z[i4 - 2 * pp - 1] = Z[i4] * (Z[i4 - 2] / Z[i4 - 2 * pp - 3]);
                d = Z[i4] * (d / Z[i4 - 2 * pp - 3]);
            }
            emin = fminf(emin, Z[i4 - 2 * pp - 1]);
        }
        Z[4 * n0 - pp - 3] = d;

        /* Now find qmax. */
        qmax = Z[4 * i0 - pp - 3];
        for (i4 = 4 * i0 - pp + 2; i4 <= 4 * n0 - pp - 2; i4 += 4) {
            qmax = fmaxf(qmax, Z[i4 - 1]);
        }

        /* Prepare for the next iteration on K. */
        pp = 1 - pp;
    }

    /*
     * Initialise variables to pass to SLASQ3.
     */
    ttype = 0;
    dmin1 = ZERO;
    dmin2 = ZERO;
    dn    = ZERO;
    dn1   = ZERO;
    dn2   = ZERO;
    g     = ZERO;
    tau   = ZERO;

    iter  = 2;
    nfail = 0;
    ndiv  = 2 * (n0 - i0);

    for (iwhila = 1; iwhila <= n + 1; iwhila++) {
        if (n0 < 1) {
            goto L170;
        }

        /*
         * While array unfinished do
         *
         * E(N0) holds the value of SIGMA when submatrix in I0:N0
         * splits from the rest of the array, but is negated.
         */
        desig = ZERO;
        if (n0 == n) {
            sigma = ZERO;
        } else {
            sigma = -Z[4 * n0 - 2];
        }
        if (sigma < ZERO) {
            *info = 1;
            return;
        }

        /*
         * Find last unreduced submatrix's top index I0, find QMAX and
         * EMIN. Find Gershgorin-type bound if Q's much greater than E's.
         */
        emax = ZERO;
        if (n0 > i0) {
            emin = fabsf(Z[4 * n0 - 6]);
        } else {
            emin = ZERO;
        }
        qmin = Z[4 * n0 - 4];
        qmax = qmin;
        for (i4 = 4 * n0; i4 >= 8; i4 -= 4) {
            if (Z[i4 - 6] <= ZERO) {
                goto L100;
            }
            if (qmin >= FOUR * emax) {
                qmin = fminf(qmin, Z[i4 - 4]);
                emax = fmaxf(emax, Z[i4 - 6]);
            }
            qmax = fmaxf(qmax, Z[i4 - 8] + Z[i4 - 6]);
            emin = fminf(emin, Z[i4 - 6]);
        }
        i4 = 4;

L100:
        i0 = i4 / 4;
        pp = 0;

        if (n0 - i0 > 1) {
            dee = Z[4 * i0 - 4];
            deemin = dee;
            kmin = i0;
            for (i4 = 4 * i0 + 1; i4 <= 4 * n0 - 3; i4 += 4) {
                dee = Z[i4 - 1] * (dee / (dee + Z[i4 - 3]));
                if (dee <= deemin) {
                    deemin = dee;
                    kmin = (i4 + 3) / 4;
                }
            }
            if ((kmin - i0) * 2 < n0 - kmin &&
                deemin <= HALF * Z[4 * n0 - 4]) {
                ipn4 = 4 * (i0 + n0);
                pp = 2;
                for (i4 = 4 * i0; i4 <= 2 * (i0 + n0 - 1); i4 += 4) {
                    temp = Z[i4 - 4];
                    Z[i4 - 4] = Z[ipn4 - i4 - 4];
                    Z[ipn4 - i4 - 4] = temp;
                    temp = Z[i4 - 3];
                    Z[i4 - 3] = Z[ipn4 - i4 - 3];
                    Z[ipn4 - i4 - 3] = temp;
                    temp = Z[i4 - 2];
                    Z[i4 - 2] = Z[ipn4 - i4 - 6];
                    Z[ipn4 - i4 - 6] = temp;
                    temp = Z[i4 - 1];
                    Z[i4 - 1] = Z[ipn4 - i4 - 5];
                    Z[ipn4 - i4 - 5] = temp;
                }
            }
        }

        /* Put -(initial shift) into DMIN. */
        dmin = -fmaxf(ZERO, qmin - TWO * sqrtf(qmin) * sqrtf(emax));

        /*
         * Now I0:N0 is unreduced.
         * PP = 0 for ping, PP = 1 for pong.
         * PP = 2 indicates that flipping was applied to the Z array and
         *        that the tests for deflation upon entry in SLASQ3
         *        should not be performed.
         */
        nbig = 100 * (n0 - i0 + 1);
        for (iwhilb = 1; iwhilb <= nbig; iwhilb++) {
            if (i0 > n0) {
                goto L150;
            }

            /* While submatrix unfinished take a good dqds step. */
            slasq3(i0, &n0, Z, &pp, &dmin, &sigma, &desig, qmax, &nfail,
                   &iter, &ndiv, ieee, &ttype, &dmin1, &dmin2, &dn, &dn1,
                   &dn2, &g, &tau);

            pp = 1 - pp;

            /* When EMIN is very small check for splits. */
            if (pp == 0 && n0 - i0 >= 3) {
                if (Z[4 * n0 - 1] <= tol2 * qmax ||
                    Z[4 * n0 - 2] <= tol2 * sigma) {
                    splt = i0 - 1;
                    qmax = Z[4 * i0 - 4];
                    emin = Z[4 * i0 - 2];
                    oldemn = Z[4 * i0 - 1];
                    for (i4 = 4 * i0; i4 <= 4 * (n0 - 3); i4 += 4) {
                        if (Z[i4 - 1] <= tol2 * Z[i4 - 4] ||
                            Z[i4 - 2] <= tol2 * sigma) {
                            Z[i4 - 2] = -sigma;
                            splt = i4 / 4;
                            qmax = ZERO;
                            emin = Z[i4 + 2];
                            oldemn = Z[i4 + 3];
                        } else {
                            qmax = fmaxf(qmax, Z[i4]);
                            emin = fminf(emin, Z[i4 - 2]);
                            oldemn = fminf(oldemn, Z[i4 - 1]);
                        }
                    }
                    Z[4 * n0 - 2] = emin;
                    Z[4 * n0 - 1] = oldemn;
                    i0 = splt + 1;
                }
            }
        }
        /* IWHILB loop exhausted */

        *info = 2;

        /*
         * Maximum number of iterations exceeded, restore the shift
         * SIGMA and place the new d's and e's in a qd array.
         * This might need to be done for several blocks.
         */
        i1 = i0;
        n1 = n0;
L145:
        tempq = Z[4 * i0 - 4];
        Z[4 * i0 - 4] = Z[4 * i0 - 4] + sigma;
        for (k = i0 + 1; k <= n0; k++) {
            tempe = Z[4 * k - 6];
            Z[4 * k - 6] = Z[4 * k - 6] * (tempq / Z[4 * k - 8]);
            tempq = Z[4 * k - 4];
            Z[4 * k - 4] = Z[4 * k - 4] + sigma + tempe - Z[4 * k - 6];
        }

        /* Prepare to do this on the previous block if there is one. */
        if (i1 > 1) {
            n1 = i1 - 1;
            while (i1 >= 2 && Z[4 * i1 - 6] >= ZERO) {
                i1 = i1 - 1;
            }
            sigma = -Z[4 * n1 - 2];
            goto L145;
        }

        for (k = 1; k <= n; k++) {
            Z[2 * k - 2] = Z[4 * k - 4];
            /*
             * Only the block 1..N0 is unfinished. The rest of the e's
             * must be essentially zero, although sometimes other data
             * has been stored in them.
             */
            if (k < n0) {
                Z[2 * k - 1] = Z[4 * k - 2];
            } else {
                Z[2 * k - 1] = 0.0f;
            }
        }
        return;

L150:
        ; /* end IWHILB: successfully converged this block */
    }

    /* IWHILA loop exhausted without convergence. */
    *info = 3;
    return;

L170:
    /*
     * Move q's to the front.
     */
    for (k = 2; k <= n; k++) {
        Z[k - 1] = Z[4 * k - 4];
    }

    /* Sort and compute sum of eigenvalues. */
    slasrt("D", n, Z, &iinfo);

    e = ZERO;
    for (k = n; k >= 1; k--) {
        e = e + Z[k - 1];
    }

    /* Store trace, sum(eigenvalues) and information on performance. */
    Z[2 * n] = trace;
    Z[2 * n + 1] = e;
    Z[2 * n + 2] = (f32)iter;
    Z[2 * n + 3] = (f32)ndiv / (f32)(n * n);
    Z[2 * n + 4] = HUNDRD * nfail / (f32)iter;
    return;
}
