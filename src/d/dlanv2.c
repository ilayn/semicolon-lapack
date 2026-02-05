/**
 * @file dlanv2.c
 * @brief DLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
 *        matrix in standard form.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
 * matrix in standard form:
 *
 *      [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ]
 *      [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ]
 *
 * where either
 * 1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or
 * 2) AA = DD and BB*CC < 0, so that AA +/- sqrt(BB*CC) are complex
 *    conjugate eigenvalues.
 *
 * @param[in,out] a      On entry, the (1,1) element of the input matrix.
 *                       On exit, the (1,1) element of the Schur form.
 * @param[in,out] b      On entry, the (1,2) element of the input matrix.
 *                       On exit, the (1,2) element of the Schur form.
 * @param[in,out] c      On entry, the (2,1) element of the input matrix.
 *                       On exit, the (2,1) element of the Schur form.
 * @param[in,out] d      On entry, the (2,2) element of the input matrix.
 *                       On exit, the (2,2) element of the Schur form.
 * @param[out]    rt1r   The real part of the first eigenvalue.
 * @param[out]    rt1i   The imaginary part of the first eigenvalue.
 * @param[out]    rt2r   The real part of the second eigenvalue.
 * @param[out]    rt2i   The imaginary part of the second eigenvalue.
 *                       If the eigenvalues are a complex conjugate pair,
 *                       RT1I > 0.
 * @param[out]    cs     Cosine of the rotation matrix.
 * @param[out]    sn     Sine of the rotation matrix.
 */
void dlanv2(double* a, double* b, double* c, double* d,
            double* rt1r, double* rt1i, double* rt2r, double* rt2i,
            double* cs, double* sn)
{
    const double ZERO = 0.0;
    const double HALF = 0.5;
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double MULTPL = 4.0;

    double aa, bb, bcmax, bcmis, cc, cs1, dd, eps, p, sab;
    double sac, scale, sigma, sn1, tau, temp, z, safmin;
    double safmn2, safmx2, base;
    int count;

    safmin = dlamch("S");
    eps = dlamch("P");
    base = dlamch("B");
    safmn2 = pow(base, (int)(log(safmin / eps) / log(base) / TWO));
    safmx2 = ONE / safmn2;

    if (*c == ZERO) {
        *cs = ONE;
        *sn = ZERO;

    } else if (*b == ZERO) {
        /* Swap rows and columns */
        *cs = ZERO;
        *sn = ONE;
        temp = *d;
        *d = *a;
        *a = temp;
        *b = -(*c);
        *c = ZERO;

    } else if ((*a - *d) == ZERO &&
               copysign(ONE, *b) != copysign(ONE, *c)) {
        /* SIGN(ONE, B) != SIGN(ONE, C) means B and C have opposite signs */
        *cs = ONE;
        *sn = ZERO;

    } else {
        temp = *a - *d;
        p = HALF * temp;
        bcmax = fabs(*b) > fabs(*c) ? fabs(*b) : fabs(*c);
        bcmis = (fabs(*b) < fabs(*c) ? fabs(*b) : fabs(*c));
        /* Apply sign of B and sign of C to bcmis: SIGN(ONE,B)*SIGN(ONE,C) */
        bcmis = copysign(bcmis, copysign(ONE, *b) * copysign(ONE, *c));
        scale = fabs(p) > bcmax ? fabs(p) : bcmax;
        z = (p / scale) * p + (bcmax / scale) * bcmis;

        /* If Z is of the order of the machine accuracy, postpone the
           decision on the nature of eigenvalues */
        if (z >= MULTPL * eps) {
            /* Real eigenvalues. Compute A and D. */
            z = p + copysign(sqrt(scale) * sqrt(z), p);
            *a = *d + z;
            *d = *d - (bcmax / z) * bcmis;

            /* Compute B and the rotation matrix */
            tau = dlapy2(*c, z);
            *cs = z / tau;
            *sn = *c / tau;
            *b = *b - *c;
            *c = ZERO;

        } else {
            /* Complex eigenvalues, or real (almost) equal eigenvalues.
               Make diagonal elements equal. */
            count = 0;
            sigma = *b + *c;
            do {
                count = count + 1;
                scale = fabs(temp) > fabs(sigma) ? fabs(temp) : fabs(sigma);
                if (scale >= safmx2) {
                    sigma = sigma * safmn2;
                    temp = temp * safmn2;
                    if (count <= 20) continue;
                }
                if (scale <= safmn2) {
                    sigma = sigma * safmx2;
                    temp = temp * safmx2;
                    if (count <= 20) continue;
                }
                break;
            } while (1);

            p = HALF * temp;
            tau = dlapy2(sigma, temp);
            *cs = sqrt(HALF * (ONE + fabs(sigma) / tau));
            *sn = -(p / (tau * (*cs))) * copysign(ONE, sigma);

            /* Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
                       [ CC  DD ]   [ C  D ] [ SN  CS ] */
            aa = (*a) * (*cs) + (*b) * (*sn);
            bb = -(*a) * (*sn) + (*b) * (*cs);
            cc = (*c) * (*cs) + (*d) * (*sn);
            dd = -(*c) * (*sn) + (*d) * (*cs);

            /* Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
                       [ C  D ]   [-SN  CS ] [ CC  DD ]

               Note: Some of the multiplications are wrapped in parentheses to
                     prevent compilers from using FMA instructions. See
                     https://github.com/Reference-LAPACK/lapack/issues/1031. */
            *a = aa * (*cs) + cc * (*sn);
            *b = (bb * (*cs)) + (dd * (*sn));
            *c = -(aa * (*sn)) + (cc * (*cs));
            *d = -bb * (*sn) + dd * (*cs);

            temp = HALF * (*a + *d);
            *a = temp;
            *d = temp;

            if (*c != ZERO) {
                if (*b != ZERO) {
                    if (copysign(ONE, *b) == copysign(ONE, *c)) {
                        /* Real eigenvalues: reduce to upper triangular form */
                        sab = sqrt(fabs(*b));
                        sac = sqrt(fabs(*c));
                        p = copysign(sab * sac, *c);
                        tau = ONE / sqrt(fabs(*b + *c));
                        *a = temp + p;
                        *d = temp - p;
                        *b = *b - *c;
                        *c = ZERO;
                        cs1 = sab * tau;
                        sn1 = sac * tau;
                        temp = (*cs) * cs1 - (*sn) * sn1;
                        *sn = (*cs) * sn1 + (*sn) * cs1;
                        *cs = temp;
                    }
                } else {
                    *b = -(*c);
                    *c = ZERO;
                    temp = *cs;
                    *cs = -(*sn);
                    *sn = temp;
                }
            }
        }
    }

    /* Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I). */
    *rt1r = *a;
    *rt2r = *d;
    if (*c == ZERO) {
        *rt1i = ZERO;
        *rt2i = ZERO;
    } else {
        *rt1i = sqrt(fabs(*b)) * sqrt(fabs(*c));
        *rt2i = -(*rt1i);
    }
}
