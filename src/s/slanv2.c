/**
 * @file slanv2.c
 * @brief SLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
 *        matrix in standard form.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
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
void slanv2(f32* a, f32* b, f32* c, f32* d,
            f32* rt1r, f32* rt1i, f32* rt2r, f32* rt2i,
            f32* cs, f32* sn)
{
    const f32 ZERO = 0.0f;
    const f32 HALF = 0.5f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 MULTPL = 4.0f;

    f32 aa, bb, bcmax, bcmis, cc, cs1, dd, eps, p, sab;
    f32 sac, scale, sigma, sn1, tau, temp, z, safmin;
    f32 safmn2, safmx2, base;
    INT count;

    safmin = slamch("S");
    eps = slamch("P");
    base = slamch("B");
    safmn2 = powf(base, (INT)(logf(safmin / eps) / logf(base) / TWO));
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
               copysignf(ONE, *b) != copysignf(ONE, *c)) {
        /* SIGN(ONE, B) != SIGN(ONE, C) means B and C have opposite signs */
        *cs = ONE;
        *sn = ZERO;

    } else {
        temp = *a - *d;
        p = HALF * temp;
        bcmax = fabsf(*b) > fabsf(*c) ? fabsf(*b) : fabsf(*c);
        bcmis = (fabsf(*b) < fabsf(*c) ? fabsf(*b) : fabsf(*c));
        /* Apply sign of B and sign of C to bcmis: SIGN(ONE,B)*SIGN(ONE,C) */
        bcmis = copysignf(bcmis, copysignf(ONE, *b) * copysignf(ONE, *c));
        scale = fabsf(p) > bcmax ? fabsf(p) : bcmax;
        z = (p / scale) * p + (bcmax / scale) * bcmis;

        /* If Z is of the order of the machine accuracy, postpone the
           decision on the nature of eigenvalues */
        if (z >= MULTPL * eps) {
            /* Real eigenvalues. Compute A and D. */
            z = p + copysignf(sqrtf(scale) * sqrtf(z), p);
            *a = *d + z;
            *d = *d - (bcmax / z) * bcmis;

            /* Compute B and the rotation matrix */
            tau = slapy2(*c, z);
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
                scale = fabsf(temp) > fabsf(sigma) ? fabsf(temp) : fabsf(sigma);
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
            tau = slapy2(sigma, temp);
            *cs = sqrtf(HALF * (ONE + fabsf(sigma) / tau));
            *sn = -(p / (tau * (*cs))) * copysignf(ONE, sigma);

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
                    if (copysignf(ONE, *b) == copysignf(ONE, *c)) {
                        /* Real eigenvalues: reduce to upper triangular form */
                        sab = sqrtf(fabsf(*b));
                        sac = sqrtf(fabsf(*c));
                        p = copysignf(sab * sac, *c);
                        tau = ONE / sqrtf(fabsf(*b + *c));
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
        *rt1i = sqrtf(fabsf(*b)) * sqrtf(fabsf(*c));
        *rt2i = -(*rt1i);
    }
}
