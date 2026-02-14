/**
 * @file slasd4.c
 * @brief SLASD4 computes the square root of the i-th updated eigenvalue of a
 *        positive symmetric rank-one modification to a positive diagonal matrix.
 */

#include "semicolon_lapack_single.h"
#include <math.h>

/* slaed6 is declared in semicolon_lapack_double.h */

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;
static const f32 TWO = 2.0f;
static const f32 THREE = 3.0f;
static const f32 FOUR = 4.0f;
static const f32 EIGHT = 8.0f;
static const f32 TEN = 10.0f;

static const int MAXIT = 400;

/**
 * SLASD4 computes the square root of the I-th updated eigenvalue of a positive
 * symmetric rank-one modification to a positive diagonal matrix whose entries
 * are given as the squares of the corresponding entries in the array D.
 *
 * The rank-one modified system is: diag(D) * diag(D) + RHO * Z * Z^T
 *
 * @param[in]     n       The length of all arrays.
 * @param[in]     i       The index of the eigenvalue to be computed. 1 <= i <= n. (1-based!)
 * @param[in]     D       Array of dimension n. The original eigenvalues, 0 <= D[j] < D[k] for j < k.
 * @param[in]     Z       Array of dimension n. The components of the updating vector.
 * @param[out]    delta   Array of dimension n. Contains (D[j] - sigma_i) in component j.
 * @param[in]     rho     The scalar in the symmetric updating formula.
 * @param[out]    sigma   The computed sigma_i, the i-th updated eigenvalue.
 * @param[out]    work    Array of dimension n. Contains (D[j] + sigma_i) in component j.
 * @param[out]    info
 *                         - = 0: successful exit. > 0: if info = 1, the updating process failed.
 */
void slasd4(const int n, const int i, const f32* const restrict D,
            const f32* const restrict Z, f32* const restrict delta,
            const f32 rho, f32* sigma, f32* const restrict work,
            int* info)
{
    /* Local variables */
    int orgati, swtch, swtch3, geomavg;
    int ii, iim1, iip1, ip1, iter, j, niter;
    f32 a, b, c, delsq, delsq2, sq2, dphi, dpsi, dtiim;
    f32 dtiip, dtipsq, dtisq, dtnsq, dtnsq1, dw, eps;
    f32 erretm, eta, phi, prew, psi, rhoinv, sglb;
    f32 sgub, tau, tau2, temp, temp1, temp2, w;
    f32 dd[3], zz[3];
    int iinfo;

    /* Quick return for n=1 and n=2 */
    *info = 0;
    if (n == 1) {
        /* Presumably i=1 upon entry */
        *sigma = sqrtf(D[0] * D[0] + rho * Z[0] * Z[0]);
        delta[0] = ONE;
        work[0] = ONE;
        return;
    }
    if (n == 2) {
        slasd5(i, D, Z, delta, rho, sigma, work);
        return;
    }

    /* Compute machine epsilon */
    eps = slamch("Epsilon");
    rhoinv = ONE / rho;
    tau2 = ZERO;

    /* Convert 1-based i to indices */
    /* The case i == n */
    if (i == n) {
        /* Initialize some basic variables */
        ii = n - 1;  /* 0-based index for n-1 */
        niter = 1;

        /* Calculate initial guess */
        temp = rho / TWO;
        temp1 = temp / (D[n - 1] + sqrtf(D[n - 1] * D[n - 1] + temp));
        for (j = 0; j < n; j++) {
            work[j] = D[j] + D[n - 1] + temp1;
            delta[j] = (D[j] - D[n - 1]) - temp1;
        }

        psi = ZERO;
        for (j = 0; j < n - 2; j++) {
            psi = psi + Z[j] * Z[j] / (delta[j] * work[j]);
        }

        c = rhoinv + psi;
        w = c + Z[ii] * Z[ii] / (delta[ii] * work[ii]) +
            Z[n - 1] * Z[n - 1] / (delta[n - 1] * work[n - 1]);

        if (w <= ZERO) {
            temp1 = sqrtf(D[n - 1] * D[n - 1] + rho);
            temp = Z[n - 2] * Z[n - 2] / ((D[n - 2] + temp1) *
                   (D[n - 1] - D[n - 2] + rho / (D[n - 1] + temp1))) +
                   Z[n - 1] * Z[n - 1] / rho;

            if (c <= temp) {
                tau = rho;
            } else {
                delsq = (D[n - 1] - D[n - 2]) * (D[n - 1] + D[n - 2]);
                a = -c * delsq + Z[n - 2] * Z[n - 2] + Z[n - 1] * Z[n - 1];
                b = Z[n - 1] * Z[n - 1] * delsq;
                if (a < ZERO) {
                    tau2 = TWO * b / (sqrtf(a * a + FOUR * b * c) - a);
                } else {
                    tau2 = (a + sqrtf(a * a + FOUR * b * c)) / (TWO * c);
                }
                tau = tau2 / (D[n - 1] + sqrtf(D[n - 1] * D[n - 1] + tau2));
            }
        } else {
            delsq = (D[n - 1] - D[n - 2]) * (D[n - 1] + D[n - 2]);
            a = -c * delsq + Z[n - 2] * Z[n - 2] + Z[n - 1] * Z[n - 1];
            b = Z[n - 1] * Z[n - 1] * delsq;

            if (a < ZERO) {
                tau2 = TWO * b / (sqrtf(a * a + FOUR * b * c) - a);
            } else {
                tau2 = (a + sqrtf(a * a + FOUR * b * c)) / (TWO * c);
            }
            tau = tau2 / (D[n - 1] + sqrtf(D[n - 1] * D[n - 1] + tau2));
        }

        *sigma = D[n - 1] + tau;
        for (j = 0; j < n; j++) {
            delta[j] = (D[j] - D[n - 1]) - tau;
            work[j] = D[j] + D[n - 1] + tau;
        }

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j < ii; j++) {
            temp = Z[j] / (delta[j] * work[j]);
            psi = psi + Z[j] * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = fabsf(erretm);

        /* Evaluate PHI and the derivative DPHI */
        temp = Z[n - 1] / (delta[n - 1] * work[n - 1]);
        phi = Z[n - 1] * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv;

        w = rhoinv + phi + psi;

        /* Test for convergence */
        if (fabsf(w) <= eps * erretm) {
            return;
        }

        /* Calculate the new step */
        niter = niter + 1;
        dtnsq1 = work[n - 2] * delta[n - 2];
        dtnsq = work[n - 1] * delta[n - 1];
        c = w - dtnsq1 * dpsi - dtnsq * dphi;
        a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
        b = dtnsq * dtnsq1 * w;
        if (c < ZERO) {
            c = fabsf(c);
        }
        if (c == ZERO) {
            eta = rho - (*sigma) * (*sigma);
        } else if (a >= ZERO) {
            eta = (a + sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
        } else {
            eta = TWO * b / (a - sqrtf(fabsf(a * a - FOUR * b * c)));
        }

        if (w * eta > ZERO) {
            eta = -w / (dpsi + dphi);
        }
        temp = eta - dtnsq;
        if (temp > rho) {
            eta = rho + dtnsq;
        }

        eta = eta / (*sigma + sqrtf(eta + (*sigma) * (*sigma)));
        tau = tau + eta;
        *sigma = *sigma + eta;

        for (j = 0; j < n; j++) {
            delta[j] = delta[j] - eta;
            work[j] = work[j] + eta;
        }

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j < ii; j++) {
            temp = Z[j] / (work[j] * delta[j]);
            psi = psi + Z[j] * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = fabsf(erretm);

        /* Evaluate PHI and the derivative DPHI */
        tau2 = work[n - 1] * delta[n - 1];
        temp = Z[n - 1] / tau2;
        phi = Z[n - 1] * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv;

        w = rhoinv + phi + psi;

        /* Main loop to update the values of the array DELTA */
        iter = niter + 1;

        for (niter = iter; niter <= MAXIT; niter++) {
            /* Test for convergence */
            if (fabsf(w) <= eps * erretm) {
                return;
            }

            /* Calculate the new step */
            dtnsq1 = work[n - 2] * delta[n - 2];
            dtnsq = work[n - 1] * delta[n - 1];
            c = w - dtnsq1 * dpsi - dtnsq * dphi;
            a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
            b = dtnsq1 * dtnsq * w;
            if (a >= ZERO) {
                eta = (a + sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
            } else {
                eta = TWO * b / (a - sqrtf(fabsf(a * a - FOUR * b * c)));
            }

            if (w * eta > ZERO) {
                eta = -w / (dpsi + dphi);
            }
            temp = eta - dtnsq;
            if (temp <= ZERO) {
                eta = eta / TWO;
            }

            eta = eta / (*sigma + sqrtf(eta + (*sigma) * (*sigma)));
            tau = tau + eta;
            *sigma = *sigma + eta;

            for (j = 0; j < n; j++) {
                delta[j] = delta[j] - eta;
                work[j] = work[j] + eta;
            }

            /* Evaluate PSI and the derivative DPSI */
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j = 0; j < ii; j++) {
                temp = Z[j] / (work[j] * delta[j]);
                psi = psi + Z[j] * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = fabsf(erretm);

            /* Evaluate PHI and the derivative DPHI */
            tau2 = work[n - 1] * delta[n - 1];
            temp = Z[n - 1] / tau2;
            phi = Z[n - 1] * temp;
            dphi = temp * temp;
            erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv;

            w = rhoinv + phi + psi;
        }

        /* Return with INFO = 1, NITER = MAXIT and not converged */
        *info = 1;
        return;

    } else {
        /* The case for i < n */
        niter = 1;
        ip1 = i;  /* 0-based index for i+1 in Fortran = i in C */

        /* Calculate initial guess */
        delsq = (D[ip1] - D[i - 1]) * (D[ip1] + D[i - 1]);
        delsq2 = delsq / TWO;
        sq2 = sqrtf((D[i - 1] * D[i - 1] + D[ip1] * D[ip1]) / TWO);
        temp = delsq2 / (D[i - 1] + sq2);
        for (j = 0; j < n; j++) {
            work[j] = D[j] + D[i - 1] + temp;
            delta[j] = (D[j] - D[i - 1]) - temp;
        }

        psi = ZERO;
        for (j = 0; j < i - 1; j++) {
            psi = psi + Z[j] * Z[j] / (work[j] * delta[j]);
        }

        phi = ZERO;
        for (j = n - 1; j >= i + 1; j--) {   /* Fortran: DO J = N, I+2, -1 */
            phi = phi + Z[j] * Z[j] / (work[j] * delta[j]);
        }
        c = rhoinv + psi + phi;
        w = c + Z[i - 1] * Z[i - 1] / (work[i - 1] * delta[i - 1]) +
            Z[ip1] * Z[ip1] / (work[ip1] * delta[ip1]);

        geomavg = 0;
        if (w > ZERO) {
            /* d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2
             * We choose d(i) as origin. */
            orgati = 1;
            ii = i - 1;  /* 0-based */
            sglb = ZERO;
            sgub = delsq2 / (D[i - 1] + sq2);
            a = c * delsq + Z[i - 1] * Z[i - 1] + Z[ip1] * Z[ip1];
            b = Z[i - 1] * Z[i - 1] * delsq;
            if (a > ZERO) {
                tau2 = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
            } else {
                tau2 = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
            }

            tau = tau2 / (D[i - 1] + sqrtf(D[i - 1] * D[i - 1] + tau2));
            temp = sqrtf(eps);
            if ((D[i - 1] <= temp * D[ip1]) && (fabsf(Z[i - 1]) <= temp) &&
                (D[i - 1] > ZERO)) {
                tau = (TEN * D[i - 1] < sgub) ? TEN * D[i - 1] : sgub;
                geomavg = 1;
            }
        } else {
            /* (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2
             * We choose d(i+1) as origin. */
            orgati = 0;
            ii = ip1;  /* 0-based */
            sglb = -delsq2 / (D[ii] + sq2);
            sgub = ZERO;
            a = c * delsq - Z[i - 1] * Z[i - 1] - Z[ip1] * Z[ip1];
            b = Z[ip1] * Z[ip1] * delsq;
            if (a < ZERO) {
                tau2 = TWO * b / (a - sqrtf(fabsf(a * a + FOUR * b * c)));
            } else {
                tau2 = -(a + sqrtf(fabsf(a * a + FOUR * b * c))) / (TWO * c);
            }

            tau = tau2 / (D[ip1] + sqrtf(fabsf(D[ip1] * D[ip1] + tau2)));
        }

        *sigma = D[ii] + tau;
        for (j = 0; j < n; j++) {
            work[j] = D[j] + D[ii] + tau;
            delta[j] = (D[j] - D[ii]) - tau;
        }
        iim1 = ii - 1;
        iip1 = ii + 1;

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j <= iim1; j++) {
            temp = Z[j] / (work[j] * delta[j]);
            psi = psi + Z[j] * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = fabsf(erretm);

        /* Evaluate PHI and the derivative DPHI */
        dphi = ZERO;
        phi = ZERO;
        for (j = n - 1; j >= iip1; j--) {
            temp = Z[j] / (work[j] * delta[j]);
            phi = phi + Z[j] * temp;
            dphi = dphi + temp * temp;
            erretm = erretm + phi;
        }

        w = rhoinv + phi + psi;

        /* W is the value of the secular function with its ii-th element removed */
        swtch3 = 0;
        if (orgati) {
            if (w < ZERO) swtch3 = 1;
        } else {
            if (w > ZERO) swtch3 = 1;
        }
        if (ii == 0 || ii == n - 1) {
            swtch3 = 0;
        }

        temp = Z[ii] / (work[ii] * delta[ii]);
        dw = dpsi + dphi + temp * temp;
        temp = Z[ii] * temp;
        w = w + temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv +
                 THREE * fabsf(temp);

        /* Test for convergence */
        if (fabsf(w) <= eps * erretm) {
            return;
        }

        if (w <= ZERO) {
            sglb = (sglb > tau) ? sglb : tau;
        } else {
            sgub = (sgub < tau) ? sgub : tau;
        }

        /* Calculate the new step */
        niter = niter + 1;
        if (!swtch3) {
            dtipsq = work[ip1] * delta[ip1];
            dtisq = work[i - 1] * delta[i - 1];
            if (orgati) {
                c = w - dtipsq * dw + delsq * (Z[i - 1] / dtisq) * (Z[i - 1] / dtisq);
            } else {
                c = w - dtisq * dw - delsq * (Z[ip1] / dtipsq) * (Z[ip1] / dtipsq);
            }
            a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
            b = dtipsq * dtisq * w;
            if (c == ZERO) {
                if (a == ZERO) {
                    if (orgati) {
                        a = Z[i - 1] * Z[i - 1] + dtipsq * dtipsq * (dpsi + dphi);
                    } else {
                        a = Z[ip1] * Z[ip1] + dtisq * dtisq * (dpsi + dphi);
                    }
                }
                eta = b / a;
            } else if (a <= ZERO) {
                eta = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
            } else {
                eta = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
            }
        } else {
            /* Interpolation using THREE most relevant poles */
            dtiim = work[iim1] * delta[iim1];
            dtiip = work[iip1] * delta[iip1];
            temp = rhoinv + psi + phi;
            if (orgati) {
                temp1 = Z[iim1] / dtiim;
                temp1 = temp1 * temp1;
                c = (temp - dtiip * (dpsi + dphi)) -
                    (D[iim1] - D[iip1]) * (D[iim1] + D[iip1]) * temp1;
                zz[0] = Z[iim1] * Z[iim1];
                if (dpsi < temp1) {
                    zz[2] = dtiip * dtiip * dphi;
                } else {
                    zz[2] = dtiip * dtiip * ((dpsi - temp1) + dphi);
                }
            } else {
                temp1 = Z[iip1] / dtiip;
                temp1 = temp1 * temp1;
                c = (temp - dtiim * (dpsi + dphi)) -
                    (D[iip1] - D[iim1]) * (D[iim1] + D[iip1]) * temp1;
                if (dphi < temp1) {
                    zz[0] = dtiim * dtiim * dpsi;
                } else {
                    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
                }
                zz[2] = Z[iip1] * Z[iip1];
            }
            zz[1] = Z[ii] * Z[ii];
            dd[0] = dtiim;
            dd[1] = delta[ii] * work[ii];
            dd[2] = dtiip;
            slaed6(niter, orgati, c, dd, zz, w, &eta, &iinfo);

            if (iinfo != 0) {
                /* If INFO is not 0, i.e., SLAED6 failed, switch back to 2 pole interpolation */
                swtch3 = 0;
                dtipsq = work[ip1] * delta[ip1];
                dtisq = work[i - 1] * delta[i - 1];
                if (orgati) {
                    c = w - dtipsq * dw + delsq * (Z[i - 1] / dtisq) * (Z[i - 1] / dtisq);
                } else {
                    c = w - dtisq * dw - delsq * (Z[ip1] / dtipsq) * (Z[ip1] / dtipsq);
                }
                a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
                b = dtipsq * dtisq * w;
                if (c == ZERO) {
                    if (a == ZERO) {
                        if (orgati) {
                            a = Z[i - 1] * Z[i - 1] + dtipsq * dtipsq * (dpsi + dphi);
                        } else {
                            a = Z[ip1] * Z[ip1] + dtisq * dtisq * (dpsi + dphi);
                        }
                    }
                    eta = b / a;
                } else if (a <= ZERO) {
                    eta = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
                } else {
                    eta = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
                }
            }
        }

        /* Note, eta should be positive if w is negative, and
         * eta should be negative otherwise. However,
         * if for some reason caused by roundoff, eta*w > 0,
         * we simply use one Newton step instead. */
        if (w * eta >= ZERO) {
            eta = -w / dw;
        }

        eta = eta / (*sigma + sqrtf((*sigma) * (*sigma) + eta));
        temp = tau + eta;
        if (temp > sgub || temp < sglb) {
            if (w < ZERO) {
                eta = (sgub - tau) / TWO;
            } else {
                eta = (sglb - tau) / TWO;
            }
            if (geomavg) {
                if (w < ZERO) {
                    if (tau > ZERO) {
                        eta = sqrtf(sgub * tau) - tau;
                    }
                } else {
                    if (sglb > ZERO) {
                        eta = sqrtf(sglb * tau) - tau;
                    }
                }
            }
        }

        prew = w;

        tau = tau + eta;
        *sigma = *sigma + eta;

        for (j = 0; j < n; j++) {
            work[j] = work[j] + eta;
            delta[j] = delta[j] - eta;
        }

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j <= iim1; j++) {
            temp = Z[j] / (work[j] * delta[j]);
            psi = psi + Z[j] * temp;
            dpsi = dpsi + temp * temp;
            erretm = erretm + psi;
        }
        erretm = fabsf(erretm);

        /* Evaluate PHI and the derivative DPHI */
        dphi = ZERO;
        phi = ZERO;
        for (j = n - 1; j >= iip1; j--) {
            temp = Z[j] / (work[j] * delta[j]);
            phi = phi + Z[j] * temp;
            dphi = dphi + temp * temp;
            erretm = erretm + phi;
        }

        tau2 = work[ii] * delta[ii];
        temp = Z[ii] / tau2;
        dw = dpsi + dphi + temp * temp;
        temp = Z[ii] * temp;
        w = rhoinv + phi + psi + temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv +
                 THREE * fabsf(temp);

        swtch = 0;
        if (orgati) {
            if (-w > fabsf(prew) / TEN) swtch = 1;
        } else {
            if (w > fabsf(prew) / TEN) swtch = 1;
        }

        /* Main loop to update the values of the array DELTA and WORK */
        iter = niter + 1;

        for (niter = iter; niter <= MAXIT; niter++) {
            /* Test for convergence */
            if (fabsf(w) <= eps * erretm) {
                return;
            }

            if (w <= ZERO) {
                sglb = (sglb > tau) ? sglb : tau;
            } else {
                sgub = (sgub < tau) ? sgub : tau;
            }

            /* Calculate the new step */
            if (!swtch3) {
                dtipsq = work[ip1] * delta[ip1];
                dtisq = work[i - 1] * delta[i - 1];
                if (!swtch) {
                    if (orgati) {
                        c = w - dtipsq * dw + delsq * (Z[i - 1] / dtisq) * (Z[i - 1] / dtisq);
                    } else {
                        c = w - dtisq * dw - delsq * (Z[ip1] / dtipsq) * (Z[ip1] / dtipsq);
                    }
                } else {
                    temp = Z[ii] / (work[ii] * delta[ii]);
                    if (orgati) {
                        dpsi = dpsi + temp * temp;
                    } else {
                        dphi = dphi + temp * temp;
                    }
                    c = w - dtisq * dpsi - dtipsq * dphi;
                }
                a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
                b = dtipsq * dtisq * w;
                if (c == ZERO) {
                    if (a == ZERO) {
                        if (!swtch) {
                            if (orgati) {
                                a = Z[i - 1] * Z[i - 1] + dtipsq * dtipsq * (dpsi + dphi);
                            } else {
                                a = Z[ip1] * Z[ip1] + dtisq * dtisq * (dpsi + dphi);
                            }
                        } else {
                            a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
                        }
                    }
                    eta = b / a;
                } else if (a <= ZERO) {
                    eta = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
                } else {
                    eta = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
                }
            } else {
                /* Interpolation using THREE most relevant poles */
                dtiim = work[iim1] * delta[iim1];
                dtiip = work[iip1] * delta[iip1];
                temp = rhoinv + psi + phi;
                if (swtch) {
                    c = temp - dtiim * dpsi - dtiip * dphi;
                    zz[0] = dtiim * dtiim * dpsi;
                    zz[2] = dtiip * dtiip * dphi;
                } else {
                    if (orgati) {
                        temp1 = Z[iim1] / dtiim;
                        temp1 = temp1 * temp1;
                        temp2 = (D[iim1] - D[iip1]) * (D[iim1] + D[iip1]) * temp1;
                        c = temp - dtiip * (dpsi + dphi) - temp2;
                        zz[0] = Z[iim1] * Z[iim1];
                        if (dpsi < temp1) {
                            zz[2] = dtiip * dtiip * dphi;
                        } else {
                            zz[2] = dtiip * dtiip * ((dpsi - temp1) + dphi);
                        }
                    } else {
                        temp1 = Z[iip1] / dtiip;
                        temp1 = temp1 * temp1;
                        temp2 = (D[iip1] - D[iim1]) * (D[iim1] + D[iip1]) * temp1;
                        c = temp - dtiim * (dpsi + dphi) - temp2;
                        if (dphi < temp1) {
                            zz[0] = dtiim * dtiim * dpsi;
                        } else {
                            zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
                        }
                        zz[2] = Z[iip1] * Z[iip1];
                    }
                }
                dd[0] = dtiim;
                dd[1] = delta[ii] * work[ii];
                dd[2] = dtiip;
                slaed6(niter, orgati, c, dd, zz, w, &eta, &iinfo);

                if (iinfo != 0) {
                    /* If INFO is not 0, switch back to two pole interpolation */
                    swtch3 = 0;
                    dtipsq = work[ip1] * delta[ip1];
                    dtisq = work[i - 1] * delta[i - 1];
                    if (!swtch) {
                        if (orgati) {
                            c = w - dtipsq * dw + delsq * (Z[i - 1] / dtisq) * (Z[i - 1] / dtisq);
                        } else {
                            c = w - dtisq * dw - delsq * (Z[ip1] / dtipsq) * (Z[ip1] / dtipsq);
                        }
                    } else {
                        temp = Z[ii] / (work[ii] * delta[ii]);
                        if (orgati) {
                            dpsi = dpsi + temp * temp;
                        } else {
                            dphi = dphi + temp * temp;
                        }
                        c = w - dtisq * dpsi - dtipsq * dphi;
                    }
                    a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
                    b = dtipsq * dtisq * w;
                    if (c == ZERO) {
                        if (a == ZERO) {
                            if (!swtch) {
                                if (orgati) {
                                    a = Z[i - 1] * Z[i - 1] + dtipsq * dtipsq * (dpsi + dphi);
                                } else {
                                    a = Z[ip1] * Z[ip1] + dtisq * dtisq * (dpsi + dphi);
                                }
                            } else {
                                a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
                            }
                        }
                        eta = b / a;
                    } else if (a <= ZERO) {
                        eta = (a - sqrtf(fabsf(a * a - FOUR * b * c))) / (TWO * c);
                    } else {
                        eta = TWO * b / (a + sqrtf(fabsf(a * a - FOUR * b * c)));
                    }
                }
            }

            /* Note, eta should be positive if w is negative, and
             * eta should be negative otherwise. */
            if (w * eta >= ZERO) {
                eta = -w / dw;
            }

            eta = eta / (*sigma + sqrtf((*sigma) * (*sigma) + eta));
            temp = tau + eta;
            if (temp > sgub || temp < sglb) {
                if (w < ZERO) {
                    eta = (sgub - tau) / TWO;
                } else {
                    eta = (sglb - tau) / TWO;
                }
                if (geomavg) {
                    if (w < ZERO) {
                        if (tau > ZERO) {
                            eta = sqrtf(sgub * tau) - tau;
                        }
                    } else {
                        if (sglb > ZERO) {
                            eta = sqrtf(sglb * tau) - tau;
                        }
                    }
                }
            }

            prew = w;

            tau = tau + eta;
            *sigma = *sigma + eta;

            for (j = 0; j < n; j++) {
                work[j] = work[j] + eta;
                delta[j] = delta[j] - eta;
            }

            /* Evaluate PSI and the derivative DPSI */
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j = 0; j <= iim1; j++) {
                temp = Z[j] / (work[j] * delta[j]);
                psi = psi + Z[j] * temp;
                dpsi = dpsi + temp * temp;
                erretm = erretm + psi;
            }
            erretm = fabsf(erretm);

            /* Evaluate PHI and the derivative DPHI */
            dphi = ZERO;
            phi = ZERO;
            for (j = n - 1; j >= iip1; j--) {
                temp = Z[j] / (work[j] * delta[j]);
                phi = phi + Z[j] * temp;
                dphi = dphi + temp * temp;
                erretm = erretm + phi;
            }

            tau2 = work[ii] * delta[ii];
            temp = Z[ii] / tau2;
            dw = dpsi + dphi + temp * temp;
            temp = Z[ii] * temp;
            w = rhoinv + phi + psi + temp;
            erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv +
                     THREE * fabsf(temp);

            if (w * prew > ZERO && fabsf(w) > fabsf(prew) / TEN) {
                swtch = !swtch;
            }
        }

        /* Return with INFO = 1, NITER = MAXIT and not converged */
        *info = 1;
    }
}
