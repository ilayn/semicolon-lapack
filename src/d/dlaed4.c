/**
 * @file dlaed4.c
 * @brief DLAED4 finds a single root of the secular equation.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAED4 computes the I-th updated eigenvalue of a symmetric
 * rank-one modification to a diagonal matrix whose elements are
 * given in the array D, and that
 *
 *            D(i) < D(j)  for  i < j
 *
 * and that RHO > 0. This is arranged by the calling routine, and is
 * no loss in generality. The rank-one modified system is thus
 *
 *            diag( D )  +  RHO * Z * Z_transpose.
 *
 * where we assume the Euclidean norm of Z is 1.
 *
 * The method consists of approximating the rational functions in the
 * secular equation by simpler interpolating rational functions.
 *
 * @param[in]     n      The length of all arrays.
 * @param[in]     i      The index of the eigenvalue to be computed. 0 <= i < n.
 * @param[in]     D      Double precision array, dimension (n).
 *                       The original eigenvalues. It is assumed that they are in
 *                       order, D(i) < D(j) for i < j.
 * @param[in]     Z      Double precision array, dimension (n).
 *                       The components of the updating vector.
 * @param[out]    delta  Double precision array, dimension (n).
 *                       Contains (D(j) - lambda_I) in its j-th component.
 * @param[in]     rho    The scalar in the symmetric updating formula.
 * @param[out]    dlam   The computed lambda_I, the I-th updated eigenvalue.
 * @param[out]    info   = 0: successful exit
 *                       > 0: if info = 1, the updating process failed.
 */
void dlaed4(const int n, const int i, const double* const restrict D,
            const double* const restrict Z, double* const restrict delta,
            const double rho, double* dlam, int* info)
{
    const int MAXIT = 30;
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double THREE = 3.0;
    const double FOUR = 4.0;
    const double EIGHT = 8.0;
    const double TEN = 10.0;

    int orgati, swtch, swtch3;
    int ii, iim1, iip1, ip1, iter, j, niter;
    double a, b, c, del, dltlb, dltub, dphi, dpsi, dw;
    double eps, erretm, eta, midpt, phi, prew, psi;
    double rhoinv, tau, temp, temp1, w;
    double zz[3];

    /* Since this routine is called in an inner loop, we do no argument
       checking. */

    *info = 0;

    /* Quick return for N=1 and 2. */
    if (n == 1) {
        /* Presumably, i=0 upon entry */
        *dlam = D[0] + rho * Z[0] * Z[0];
        delta[0] = ONE;
        return;
    }
    if (n == 2) {
        dlaed5(i, D, Z, delta, rho, dlam);
        return;
    }

    /* Compute machine epsilon */
    eps = dlamch("E");
    rhoinv = ONE / rho;

    /* The case i = n-1 (last eigenvalue) */
    if (i == n - 1) {

        /* Initialize some basic variables */
        ii = n - 2;
        niter = 1;

        /* Calculate initial guess */
        midpt = rho / TWO;

        /* If ||Z||_2 is not one, then TEMP should be set to
           RHO * ||Z||_2^2 / TWO */
        for (j = 0; j < n; j++) {
            delta[j] = (D[j] - D[i]) - midpt;
        }

        psi = ZERO;
        for (j = 0; j <= n - 3; j++) {
            psi += Z[j] * Z[j] / delta[j];
        }

        c = rhoinv + psi;
        w = c + Z[ii] * Z[ii] / delta[ii] +
            Z[n - 1] * Z[n - 1] / delta[n - 1];

        if (w <= ZERO) {
            temp = Z[n - 2] * Z[n - 2] / (D[n - 1] - D[n - 2] + rho) +
                   Z[n - 1] * Z[n - 1] / rho;
            if (c <= temp) {
                tau = rho;
            } else {
                del = D[n - 1] - D[n - 2];
                a = -c * del + Z[n - 2] * Z[n - 2] + Z[n - 1] * Z[n - 1];
                b = Z[n - 1] * Z[n - 1] * del;
                if (a < ZERO) {
                    tau = TWO * b / (sqrt(a * a + FOUR * b * c) - a);
                } else {
                    tau = (a + sqrt(a * a + FOUR * b * c)) / (TWO * c);
                }
            }

            /* It can be proved that
               D(n-1)+RHO/2 <= LAMBDA(n-1) < D(n-1)+TAU <= D(n-1)+RHO */
            dltlb = midpt;
            dltub = rho;
        } else {
            del = D[n - 1] - D[n - 2];
            a = -c * del + Z[n - 2] * Z[n - 2] + Z[n - 1] * Z[n - 1];
            b = Z[n - 1] * Z[n - 1] * del;
            if (a < ZERO) {
                tau = TWO * b / (sqrt(a * a + FOUR * b * c) - a);
            } else {
                tau = (a + sqrt(a * a + FOUR * b * c)) / (TWO * c);
            }

            /* It can be proved that
               D(n-1) < D(n-1)+TAU < LAMBDA(n-1) < D(n-1)+RHO/2 */
            dltlb = ZERO;
            dltub = midpt;
        }

        for (j = 0; j < n; j++) {
            delta[j] = (D[j] - D[i]) - tau;
        }

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j <= ii; j++) {
            temp = Z[j] / delta[j];
            psi += Z[j] * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = fabs(erretm);

        /* Evaluate PHI and the derivative DPHI */
        temp = Z[n - 1] / delta[n - 1];
        phi = Z[n - 1] * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv +
                 fabs(tau) * (dpsi + dphi);

        w = rhoinv + phi + psi;

        /* Test for convergence */
        if (fabs(w) <= eps * erretm) {
            *dlam = D[i] + tau;
            return;
        }

        if (w <= ZERO) {
            dltlb = fmax(dltlb, tau);
        } else {
            dltub = fmin(dltub, tau);
        }

        /* Calculate the new step */
        niter++;
        c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
        a = (delta[n - 2] + delta[n - 1]) * w -
            delta[n - 2] * delta[n - 1] * (dpsi + dphi);
        b = delta[n - 2] * delta[n - 1] * w;
        if (c < ZERO) c = fabs(c);
        if (c == ZERO) {
            /* Update proposed by Li, Ren-Cang: */
            eta = -w / (dpsi + dphi);
        } else if (a >= ZERO) {
            eta = (a + sqrt(fabs(a * a - FOUR * b * c))) / (TWO * c);
        } else {
            eta = TWO * b / (a - sqrt(fabs(a * a - FOUR * b * c)));
        }

        /* Note, eta should be positive if w is negative, and
           eta should be negative otherwise. However,
           if for some reason caused by roundoff, eta*w > 0,
           we simply use one Newton step instead. This way
           will guarantee eta*w < 0. */
        if (w * eta > ZERO)
            eta = -w / (dpsi + dphi);
        temp = tau + eta;
        if (temp > dltub || temp < dltlb) {
            if (w < ZERO) {
                eta = (dltub - tau) / TWO;
            } else {
                eta = (dltlb - tau) / TWO;
            }
        }
        for (j = 0; j < n; j++) {
            delta[j] -= eta;
        }

        tau += eta;

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j <= ii; j++) {
            temp = Z[j] / delta[j];
            psi += Z[j] * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = fabs(erretm);

        /* Evaluate PHI and the derivative DPHI */
        temp = Z[n - 1] / delta[n - 1];
        phi = Z[n - 1] * temp;
        dphi = temp * temp;
        erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv +
                 fabs(tau) * (dpsi + dphi);

        w = rhoinv + phi + psi;

        /* Main loop to update the values of the array DELTA */
        iter = niter + 1;

        for (niter = iter; niter <= MAXIT; niter++) {

            /* Test for convergence */
            if (fabs(w) <= eps * erretm) {
                *dlam = D[i] + tau;
                return;
            }

            if (w <= ZERO) {
                dltlb = fmax(dltlb, tau);
            } else {
                dltub = fmin(dltub, tau);
            }

            /* Calculate the new step */
            c = w - delta[n - 2] * dpsi - delta[n - 1] * dphi;
            a = (delta[n - 2] + delta[n - 1]) * w -
                delta[n - 2] * delta[n - 1] * (dpsi + dphi);
            b = delta[n - 2] * delta[n - 1] * w;
            if (a >= ZERO) {
                eta = (a + sqrt(fabs(a * a - FOUR * b * c))) / (TWO * c);
            } else {
                eta = TWO * b / (a - sqrt(fabs(a * a - FOUR * b * c)));
            }

            /* Note, eta should be positive if w is negative, and
               eta should be negative otherwise. However,
               if for some reason caused by roundoff, eta*w > 0,
               we simply use one Newton step instead. This way
               will guarantee eta*w < 0. */
            if (w * eta > ZERO)
                eta = -w / (dpsi + dphi);
            temp = tau + eta;
            if (temp > dltub || temp < dltlb) {
                if (w < ZERO) {
                    eta = (dltub - tau) / TWO;
                } else {
                    eta = (dltlb - tau) / TWO;
                }
            }
            for (j = 0; j < n; j++) {
                delta[j] -= eta;
            }

            tau += eta;

            /* Evaluate PSI and the derivative DPSI */
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j = 0; j <= ii; j++) {
                temp = Z[j] / delta[j];
                psi += Z[j] * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = fabs(erretm);

            /* Evaluate PHI and the derivative DPHI */
            temp = Z[n - 1] / delta[n - 1];
            phi = Z[n - 1] * temp;
            dphi = temp * temp;
            erretm = EIGHT * (-phi - psi) + erretm - phi + rhoinv +
                     fabs(tau) * (dpsi + dphi);

            w = rhoinv + phi + psi;
        }

        /* Return with INFO = 1, NITER = MAXIT and not converged */
        *info = 1;
        *dlam = D[i] + tau;
        return;

        /* End for the case i = n-1 */

    } else {

        /* The case for i < n-1 */
        niter = 1;
        ip1 = i + 1;

        /* Calculate initial guess */
        del = D[ip1] - D[i];
        midpt = del / TWO;
        for (j = 0; j < n; j++) {
            delta[j] = (D[j] - D[i]) - midpt;
        }

        psi = ZERO;
        for (j = 0; j <= i - 1; j++) {
            psi += Z[j] * Z[j] / delta[j];
        }

        phi = ZERO;
        for (j = n - 1; j >= i + 2; j--) {
            phi += Z[j] * Z[j] / delta[j];
        }
        c = rhoinv + psi + phi;
        w = c + Z[i] * Z[i] / delta[i] +
            Z[ip1] * Z[ip1] / delta[ip1];

        if (w > ZERO) {

            /* D(i) < the ith eigenvalue < (D(i)+D(i+1))/2

               We choose D(i) as origin. */
            orgati = 1;
            a = c * del + Z[i] * Z[i] + Z[ip1] * Z[ip1];
            b = Z[i] * Z[i] * del;
            if (a > ZERO) {
                tau = TWO * b / (a + sqrt(fabs(a * a - FOUR * b * c)));
            } else {
                tau = (a - sqrt(fabs(a * a - FOUR * b * c))) / (TWO * c);
            }
            dltlb = ZERO;
            dltub = midpt;
        } else {

            /* (D(i)+D(i+1))/2 <= the ith eigenvalue < D(i+1)

               We choose D(i+1) as origin. */
            orgati = 0;
            a = c * del - Z[i] * Z[i] - Z[ip1] * Z[ip1];
            b = Z[ip1] * Z[ip1] * del;
            if (a < ZERO) {
                tau = TWO * b / (a - sqrt(fabs(a * a + FOUR * b * c)));
            } else {
                tau = -(a + sqrt(fabs(a * a + FOUR * b * c))) / (TWO * c);
            }
            dltlb = -midpt;
            dltub = ZERO;
        }

        if (orgati) {
            for (j = 0; j < n; j++) {
                delta[j] = (D[j] - D[i]) - tau;
            }
        } else {
            for (j = 0; j < n; j++) {
                delta[j] = (D[j] - D[ip1]) - tau;
            }
        }
        if (orgati) {
            ii = i;
        } else {
            ii = i + 1;
        }
        iim1 = ii - 1;
        iip1 = ii + 1;

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j <= iim1; j++) {
            temp = Z[j] / delta[j];
            psi += Z[j] * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = fabs(erretm);

        /* Evaluate PHI and the derivative DPHI */
        dphi = ZERO;
        phi = ZERO;
        for (j = n - 1; j >= iip1; j--) {
            temp = Z[j] / delta[j];
            phi += Z[j] * temp;
            dphi += temp * temp;
            erretm += phi;
        }

        w = rhoinv + phi + psi;

        /* W is the value of the secular function with
           its ii-th element removed. */
        swtch3 = 0;
        if (orgati) {
            if (w < ZERO) swtch3 = 1;
        } else {
            if (w > ZERO) swtch3 = 1;
        }
        if (ii == 0 || ii == n - 1)
            swtch3 = 0;

        temp = Z[ii] / delta[ii];
        dw = dpsi + dphi + temp * temp;
        temp = Z[ii] * temp;
        w += temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv +
                 THREE * fabs(temp) + fabs(tau) * dw;

        /* Test for convergence */
        if (fabs(w) <= eps * erretm) {
            if (orgati) {
                *dlam = D[i] + tau;
            } else {
                *dlam = D[ip1] + tau;
            }
            return;
        }

        if (w <= ZERO) {
            dltlb = fmax(dltlb, tau);
        } else {
            dltub = fmin(dltub, tau);
        }

        /* Calculate the new step */
        niter++;
        if (!swtch3) {
            if (orgati) {
                c = w - delta[ip1] * dw - (D[i] - D[ip1]) *
                    (Z[i] / delta[i]) * (Z[i] / delta[i]);
            } else {
                c = w - delta[i] * dw - (D[ip1] - D[i]) *
                    (Z[ip1] / delta[ip1]) * (Z[ip1] / delta[ip1]);
            }
            a = (delta[i] + delta[ip1]) * w -
                delta[i] * delta[ip1] * dw;
            b = delta[i] * delta[ip1] * w;
            if (c == ZERO) {
                if (a == ZERO) {
                    if (orgati) {
                        a = Z[i] * Z[i] + delta[ip1] * delta[ip1] *
                            (dpsi + dphi);
                    } else {
                        a = Z[ip1] * Z[ip1] + delta[i] * delta[i] *
                            (dpsi + dphi);
                    }
                }
                eta = b / a;
            } else if (a <= ZERO) {
                eta = (a - sqrt(fabs(a * a - FOUR * b * c))) / (TWO * c);
            } else {
                eta = TWO * b / (a + sqrt(fabs(a * a - FOUR * b * c)));
            }
        } else {

            /* Interpolation using THREE most relevant poles */
            temp = rhoinv + psi + phi;
            if (orgati) {
                temp1 = Z[iim1] / delta[iim1];
                temp1 = temp1 * temp1;
                c = temp - delta[iip1] * (dpsi + dphi) -
                    (D[iim1] - D[iip1]) * temp1;
                zz[0] = Z[iim1] * Z[iim1];
                zz[2] = delta[iip1] * delta[iip1] *
                         ((dpsi - temp1) + dphi);
            } else {
                temp1 = Z[iip1] / delta[iip1];
                temp1 = temp1 * temp1;
                c = temp - delta[iim1] * (dpsi + dphi) -
                    (D[iip1] - D[iim1]) * temp1;
                zz[0] = delta[iim1] * delta[iim1] *
                         (dpsi + (dphi - temp1));
                zz[2] = Z[iip1] * Z[iip1];
            }
            zz[1] = Z[ii] * Z[ii];
            dlaed6(niter, orgati, c, &delta[iim1], zz, w, &eta, info);
            if (*info != 0)
                return;
        }

        /* Note, eta should be positive if w is negative, and
           eta should be negative otherwise. However,
           if for some reason caused by roundoff, eta*w > 0,
           we simply use one Newton step instead. This way
           will guarantee eta*w < 0. */
        if (w * eta >= ZERO)
            eta = -w / dw;
        temp = tau + eta;
        if (temp > dltub || temp < dltlb) {
            if (w < ZERO) {
                eta = (dltub - tau) / TWO;
            } else {
                eta = (dltlb - tau) / TWO;
            }
        }

        prew = w;

        for (j = 0; j < n; j++) {
            delta[j] -= eta;
        }

        /* Evaluate PSI and the derivative DPSI */
        dpsi = ZERO;
        psi = ZERO;
        erretm = ZERO;
        for (j = 0; j <= iim1; j++) {
            temp = Z[j] / delta[j];
            psi += Z[j] * temp;
            dpsi += temp * temp;
            erretm += psi;
        }
        erretm = fabs(erretm);

        /* Evaluate PHI and the derivative DPHI */
        dphi = ZERO;
        phi = ZERO;
        for (j = n - 1; j >= iip1; j--) {
            temp = Z[j] / delta[j];
            phi += Z[j] * temp;
            dphi += temp * temp;
            erretm += phi;
        }

        temp = Z[ii] / delta[ii];
        dw = dpsi + dphi + temp * temp;
        temp = Z[ii] * temp;
        w = rhoinv + phi + psi + temp;
        erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv +
                 THREE * fabs(temp) + fabs(tau + eta) * dw;

        swtch = 0;
        if (orgati) {
            if (-w > fabs(prew) / TEN) swtch = 1;
        } else {
            if (w > fabs(prew) / TEN) swtch = 1;
        }

        tau += eta;

        /* Main loop to update the values of the array DELTA */
        iter = niter + 1;

        for (niter = iter; niter <= MAXIT; niter++) {

            /* Test for convergence */
            if (fabs(w) <= eps * erretm) {
                if (orgati) {
                    *dlam = D[i] + tau;
                } else {
                    *dlam = D[ip1] + tau;
                }
                return;
            }

            if (w <= ZERO) {
                dltlb = fmax(dltlb, tau);
            } else {
                dltub = fmin(dltub, tau);
            }

            /* Calculate the new step */
            if (!swtch3) {
                if (!swtch) {
                    if (orgati) {
                        c = w - delta[ip1] * dw -
                            (D[i] - D[ip1]) * (Z[i] / delta[i]) * (Z[i] / delta[i]);
                    } else {
                        c = w - delta[i] * dw - (D[ip1] - D[i]) *
                            (Z[ip1] / delta[ip1]) * (Z[ip1] / delta[ip1]);
                    }
                } else {
                    temp = Z[ii] / delta[ii];
                    if (orgati) {
                        dpsi += temp * temp;
                    } else {
                        dphi += temp * temp;
                    }
                    c = w - delta[i] * dpsi - delta[ip1] * dphi;
                }
                a = (delta[i] + delta[ip1]) * w -
                    delta[i] * delta[ip1] * dw;
                b = delta[i] * delta[ip1] * w;
                if (c == ZERO) {
                    if (a == ZERO) {
                        if (!swtch) {
                            if (orgati) {
                                a = Z[i] * Z[i] + delta[ip1] *
                                    delta[ip1] * (dpsi + dphi);
                            } else {
                                a = Z[ip1] * Z[ip1] +
                                    delta[i] * delta[i] * (dpsi + dphi);
                            }
                        } else {
                            a = delta[i] * delta[i] * dpsi +
                                delta[ip1] * delta[ip1] * dphi;
                        }
                    }
                    eta = b / a;
                } else if (a <= ZERO) {
                    eta = (a - sqrt(fabs(a * a - FOUR * b * c))) / (TWO * c);
                } else {
                    eta = TWO * b / (a + sqrt(fabs(a * a - FOUR * b * c)));
                }
            } else {

                /* Interpolation using THREE most relevant poles */
                temp = rhoinv + psi + phi;
                if (swtch) {
                    c = temp - delta[iim1] * dpsi - delta[iip1] * dphi;
                    zz[0] = delta[iim1] * delta[iim1] * dpsi;
                    zz[2] = delta[iip1] * delta[iip1] * dphi;
                } else {
                    if (orgati) {
                        temp1 = Z[iim1] / delta[iim1];
                        temp1 = temp1 * temp1;
                        c = temp - delta[iip1] * (dpsi + dphi) -
                            (D[iim1] - D[iip1]) * temp1;
                        zz[0] = Z[iim1] * Z[iim1];
                        zz[2] = delta[iip1] * delta[iip1] *
                                 ((dpsi - temp1) + dphi);
                    } else {
                        temp1 = Z[iip1] / delta[iip1];
                        temp1 = temp1 * temp1;
                        c = temp - delta[iim1] * (dpsi + dphi) -
                            (D[iip1] - D[iim1]) * temp1;
                        zz[0] = delta[iim1] * delta[iim1] *
                                 (dpsi + (dphi - temp1));
                        zz[2] = Z[iip1] * Z[iip1];
                    }
                }
                zz[1] = Z[ii] * Z[ii];
                dlaed6(niter, orgati, c, &delta[iim1], zz, w, &eta, info);
                if (*info != 0)
                    return;
            }

            /* Note, eta should be positive if w is negative, and
               eta should be negative otherwise. However,
               if for some reason caused by roundoff, eta*w > 0,
               we simply use one Newton step instead. This way
               will guarantee eta*w < 0. */
            if (w * eta >= ZERO)
                eta = -w / dw;
            temp = tau + eta;
            if (temp > dltub || temp < dltlb) {
                if (w < ZERO) {
                    eta = (dltub - tau) / TWO;
                } else {
                    eta = (dltlb - tau) / TWO;
                }
            }

            for (j = 0; j < n; j++) {
                delta[j] -= eta;
            }

            tau += eta;
            prew = w;

            /* Evaluate PSI and the derivative DPSI */
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j = 0; j <= iim1; j++) {
                temp = Z[j] / delta[j];
                psi += Z[j] * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = fabs(erretm);

            /* Evaluate PHI and the derivative DPHI */
            dphi = ZERO;
            phi = ZERO;
            for (j = n - 1; j >= iip1; j--) {
                temp = Z[j] / delta[j];
                phi += Z[j] * temp;
                dphi += temp * temp;
                erretm += phi;
            }

            temp = Z[ii] / delta[ii];
            dw = dpsi + dphi + temp * temp;
            temp = Z[ii] * temp;
            w = rhoinv + phi + psi + temp;
            erretm = EIGHT * (phi - psi) + erretm + TWO * rhoinv +
                     THREE * fabs(temp) + fabs(tau) * dw;
            if (w * prew > ZERO && fabs(w) > fabs(prew) / TEN)
                swtch = !swtch;
        }

        /* Return with INFO = 1, NITER = MAXIT and not converged */
        *info = 1;
        if (orgati) {
            *dlam = D[i] + tau;
        } else {
            *dlam = D[ip1] + tau;
        }
    }

}
