/**
 * @file slaed6.c
 * @brief SLAED6 computes one Newton step in solution of the secular equation.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAED6 computes the positive or negative root (closest to the origin)
 * of
 *                  z(0)        z(1)        z(2)
 * f(x) =   rho + --------- + ---------- + ---------
 *                 d(0)-x      d(1)-x      d(2)-x
 *
 * It is assumed that
 *
 *       if orgati is true the root is between d[1] and d[2];
 *       otherwise it is between d[0] and d[1]
 *
 * This routine will be called by SLAED4 when necessary. In most cases,
 * the root sought is the smallest in magnitude, though it might not be
 * in some extremely rare situations.
 *
 * @param[in]     kniter Refer to SLAED4 for its significance.
 * @param[in]     orgati If orgati is nonzero (true), the needed root is between
 *                       d[1] and d[2]; otherwise it is between d[0] and d[1].
 *                       See SLAED4 for further details.
 * @param[in]     rho    The scalar in the equation f(x) above.
 * @param[in]     D      Double precision array, dimension (3).
 *                       D satisfies d[0] < d[1] < d[2].
 * @param[in]     Z      Double precision array, dimension (3).
 *                       Each of the elements in Z must be positive.
 * @param[in]     finit  The value of f at 0. It is more accurate than the one
 *                       evaluated inside this routine.
 * @param[out]    tau    The root of the equation f(x).
 * @param[out]    info   = 0: successful exit
 *                       > 0: if info = 1, failure to converge
 */
void slaed6(const int kniter, const int orgati, const float rho,
            const float* const restrict D, const float* const restrict Z,
            const float finit, float* tau, int* info)
{
    const int MAXIT = 40;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float THREE = 3.0f;
    const float FOUR = 4.0f;
    const float EIGHT = 8.0f;

    float dscale[3], zscale[3];

    int i, niter, scale;
    float a, b, base, c, ddf, df, eps, erretm, eta, f,
           fc, sclfac, sclinv, small1, small2, sminv1,
           sminv2, temp, temp1, temp2, temp3, temp4,
           lbd, ubd;

    *info = 0;

    if (orgati) {
        lbd = D[1];
        ubd = D[2];
    } else {
        lbd = D[0];
        ubd = D[1];
    }
    if (finit < ZERO) {
        lbd = ZERO;
    } else {
        ubd = ZERO;
    }

    niter = 1;
    *tau = ZERO;
    if (kniter == 2) {
        if (orgati) {
            temp = (D[2] - D[1]) / TWO;
            c = rho + Z[0] / ((D[0] - D[1]) - temp);
            a = c * (D[1] + D[2]) + Z[1] + Z[2];
            b = c * D[1] * D[2] + Z[1] * D[2] + Z[2] * D[1];
        } else {
            temp = (D[0] - D[1]) / TWO;
            c = rho + Z[2] / ((D[2] - D[1]) - temp);
            a = c * (D[0] + D[1]) + Z[0] + Z[1];
            b = c * D[0] * D[1] + Z[0] * D[1] + Z[1] * D[0];
        }
        temp = fmaxf(fabsf(a), fmaxf(fabsf(b), fabsf(c)));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if (c == ZERO) {
            *tau = b / a;
        } else if (a <= ZERO) {
            *tau = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
        } else {
            *tau = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
        }
        if (*tau < lbd || *tau > ubd) {
            *tau = (lbd + ubd) / TWO;
        }
        if (D[0] == *tau || D[1] == *tau || D[2] == *tau) {
            *tau = ZERO;
        } else {
            temp = finit + (*tau) * Z[0] / (D[0] * (D[0] - *tau)) +
                           (*tau) * Z[1] / (D[1] * (D[1] - *tau)) +
                           (*tau) * Z[2] / (D[2] * (D[2] - *tau));
            if (temp <= ZERO) {
                lbd = *tau;
            } else {
                ubd = *tau;
            }
            if (fabsf(finit) <= fabsf(temp)) {
                *tau = ZERO;
            }
        }
    }

    /* Get machine parameters for possible scaling to avoid overflow */

    /* Modified by Sven: parameters SMALL1, SMINV1, SMALL2,
       SMINV2, EPS are not SAVEd anymore between one call to the
       others but recomputed at each call */

    eps = slamch("E");
    base = slamch("B");
    small1 = powf(base, (int)(logf(slamch("S")) / logf(base) / THREE));
    sminv1 = ONE / small1;
    small2 = small1 * small1;
    sminv2 = sminv1 * sminv1;

    /* Determine if scaling of inputs necessary to avoid overflow
       when computing 1/TEMP**3 */

    if (orgati) {
        temp = fminf(fabsf(D[1] - *tau), fabsf(D[2] - *tau));
    } else {
        temp = fminf(fabsf(D[0] - *tau), fabsf(D[1] - *tau));
    }
    scale = 0;
    if (temp <= small1) {
        scale = 1;
        if (temp <= small2) {
            /* Scale up by power of radix nearest 1/SAFMIN**(2/3) */
            sclfac = sminv2;
            sclinv = small2;
        } else {
            /* Scale up by power of radix nearest 1/SAFMIN**(1/3) */
            sclfac = sminv1;
            sclinv = small1;
        }

        /* Scaling up safe because D, Z, TAU scaled elsewhere to be O(1) */
        for (i = 0; i < 3; i++) {
            dscale[i] = D[i] * sclfac;
            zscale[i] = Z[i] * sclfac;
        }
        *tau = (*tau) * sclfac;
        lbd = lbd * sclfac;
        ubd = ubd * sclfac;
    } else {
        /* Copy D and Z to DSCALE and ZSCALE */
        for (i = 0; i < 3; i++) {
            dscale[i] = D[i];
            zscale[i] = Z[i];
        }
    }

    fc = ZERO;
    df = ZERO;
    ddf = ZERO;
    for (i = 0; i < 3; i++) {
        temp = ONE / (dscale[i] - *tau);
        temp1 = zscale[i] * temp;
        temp2 = temp1 * temp;
        temp3 = temp2 * temp;
        fc = fc + temp1 / dscale[i];
        df = df + temp2;
        ddf = ddf + temp3;
    }
    f = finit + (*tau) * fc;

    if (fabsf(f) <= ZERO) {
        goto done;
    }
    if (f <= ZERO) {
        lbd = *tau;
    } else {
        ubd = *tau;
    }

    /* Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
       scheme

       It is not hard to see that

           1) Iterations will go up monotonically
              if FINIT < 0;

           2) Iterations will go down monotonically
              if FINIT > 0. */

    for (niter = niter + 1; niter <= MAXIT; niter++) {

        if (orgati) {
            temp1 = dscale[1] - *tau;
            temp2 = dscale[2] - *tau;
        } else {
            temp1 = dscale[0] - *tau;
            temp2 = dscale[1] - *tau;
        }
        a = (temp1 + temp2) * f - temp1 * temp2 * df;
        b = temp1 * temp2 * f;
        c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
        temp = fmaxf(fabsf(a), fmaxf(fabsf(b), fabsf(c)));
        a = a / temp;
        b = b / temp;
        c = c / temp;
        if (c == ZERO) {
            eta = b / a;
        } else if (a <= ZERO) {
            eta = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
        } else {
            eta = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
        }
        if (f * eta >= ZERO) {
            eta = -f / df;
        }

        *tau = *tau + eta;
        if (*tau < lbd || *tau > ubd) {
            *tau = (lbd + ubd) / TWO;
        }

        fc = ZERO;
        erretm = ZERO;
        df = ZERO;
        ddf = ZERO;
        for (i = 0; i < 3; i++) {
            if ((dscale[i] - *tau) != ZERO) {
                temp = ONE / (dscale[i] - *tau);
                temp1 = zscale[i] * temp;
                temp2 = temp1 * temp;
                temp3 = temp2 * temp;
                temp4 = temp1 / dscale[i];
                fc = fc + temp4;
                erretm = erretm + fabsf(temp4);
                df = df + temp2;
                ddf = ddf + temp3;
            } else {
                goto done;
            }
        }
        f = finit + (*tau) * fc;
        erretm = EIGHT * (fabsf(finit) + fabsf(*tau) * erretm) +
                 fabsf(*tau) * df;
        if ((fabsf(f) <= FOUR * eps * erretm) ||
            ((ubd - lbd) <= FOUR * eps * fabsf(*tau))) {
            goto done;
        }
        if (f <= ZERO) {
            lbd = *tau;
        } else {
            ubd = *tau;
        }
    }
    *info = 1;

done:
    /* Undo scaling */
    if (scale) {
        *tau = (*tau) * sclinv;
    }
}
