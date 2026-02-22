/**
 * @file zlargv.c
 * @brief ZLARGV generates a vector of plane rotations with real cosines and complex sines.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/** @cond */
static inline f64 abs1(c128 ff) {
    f64 a = creal(ff);
    f64 b = cimag(ff);
    return (fabs(a) > fabs(b)) ? fabs(a) : fabs(b);
}

static inline f64 abssq(c128 ff) {
    return creal(ff) * creal(ff) + cimag(ff) * cimag(ff);
}
/** @endcond */

/**
 * ZLARGV generates a vector of complex plane rotations with real
 * cosines, determined by elements of the complex vectors x and y.
 * For i = 0,1,...,n-1
 *
 *    (        c(i)   s(i) ) ( x(i) ) = ( r(i) )
 *    ( -conjg(s(i))  c(i) ) ( y(i) ) = (   0  )
 *
 *    where c(i)**2 + ABS(s(i))**2 = 1
 *
 * The following conventions are used (these are the same as in ZLARTG,
 * but differ from the BLAS1 routine ZROTG):
 *    If y(i)=0, then c(i)=1 and s(i)=0.
 *    If x(i)=0, then c(i)=0 and s(i) is chosen so that r(i) is real.
 *
 * @param[in]     n     The number of plane rotations to be generated.
 * @param[in,out] X     Double complex array, dimension (1+(n-1)*incx).
 *                      On entry, the vector x.
 *                      On exit, x(i) is overwritten by r(i), for i = 0,...,n-1.
 * @param[in]     incx  The increment between elements of X. incx > 0.
 * @param[in,out] Y     Double complex array, dimension (1+(n-1)*incy).
 *                      On entry, the vector y.
 *                      On exit, the sines of the plane rotations.
 * @param[in]     incy  The increment between elements of Y. incy > 0.
 * @param[out]    C     Double precision array, dimension (1+(n-1)*incc).
 *                      The cosines of the plane rotations.
 * @param[in]     incc  The increment between elements of C. incc > 0.
 */
void zlargv(const INT n, c128* restrict X, const INT incx,
            c128* restrict Y, const INT incy,
            f64* restrict C, const INT incc)
{
    const f64 two = 2.0;
    const f64 one = 1.0;
    const f64 zero = 0.0;
    const c128 czero = CMPLX(0.0, 0.0);

    f64 safmin = dlamch("S");
    f64 eps = dlamch("E");
    f64 safmn2 = pow(dlamch("B"),
                        (INT)(log(safmin / eps) / log(dlamch("B")) / two));
    f64 safmx2 = one / safmn2;

    INT ix = 0;
    INT iy = 0;
    INT ic = 0;

    for (INT i = 0; i < n; i++) {
        c128 f = X[ix];
        c128 g = Y[iy];

        f64 scale = (abs1(f) > abs1(g)) ? abs1(f) : abs1(g);
        c128 fs = f;
        c128 gs = g;
        INT count = 0;
        f64 cs;
        c128 sn, r;
        f64 f2, g2;

        if (scale >= safmx2) {
            do {
                count = count + 1;
                fs = fs * safmn2;
                gs = gs * safmn2;
                scale = scale * safmn2;
            } while (scale >= safmx2 && count < 20);
        } else if (scale <= safmn2) {
            if (g == czero) {
                cs = one;
                sn = czero;
                r = f;
                goto label50;
            }
            do {
                count = count - 1;
                fs = fs * safmx2;
                gs = gs * safmx2;
                scale = scale * safmx2;
            } while (scale <= safmn2);
        }

        f2 = abssq(fs);
        g2 = abssq(gs);

        if (f2 <= ((g2 > one) ? g2 : one) * safmin) {

            /* This is a rare case: F is very small. */

            if (f == czero) {
                cs = zero;
                r = dlapy2(creal(g), cimag(g));
                /* Do complex/real division explicitly with two real
                   divisions */
                f64 d = dlapy2(creal(gs), cimag(gs));
                sn = CMPLX(creal(gs) / d, -cimag(gs) / d);
                goto label50;
            }
            f64 f2s = dlapy2(creal(fs), cimag(fs));
            /* G2 and G2S are accurate
               G2 is at least SAFMIN, and G2S is at least SAFMN2 */
            f64 g2s = sqrt(g2);
            /* Error in CS from underflow in F2S is at most
               UNFL / SAFMN2 .lt. sqrt(UNFL*EPS) .lt. EPS
               If MAX(G2,ONE)=G2, then F2 .lt. G2*SAFMIN,
               and so CS .lt. sqrt(SAFMIN)
               If MAX(G2,ONE)=ONE, then F2 .lt. SAFMIN
               and so CS .lt. sqrt(SAFMIN)/SAFMN2 = sqrt(EPS)
               Therefore, CS = F2S/G2S / sqrt( 1 + (F2S/G2S)**2 ) = F2S/G2S */
            cs = f2s / g2s;
            /* Make sure abs(FF) = 1
               Do complex/real division explicitly with 2 real divisions */
            c128 ff;
            f64 d;
            if (abs1(f) > one) {
                d = dlapy2(creal(f), cimag(f));
                ff = CMPLX(creal(f) / d, cimag(f) / d);
            } else {
                f64 dr = safmx2 * creal(f);
                f64 di = safmx2 * cimag(f);
                d = dlapy2(dr, di);
                ff = CMPLX(dr / d, di / d);
            }
            sn = ff * CMPLX(creal(gs) / g2s, -cimag(gs) / g2s);
            r = cs * f + sn * g;
        } else {

            /* This is the most common case.
               Neither F2 nor F2/G2 are less than SAFMIN
               F2S cannot overflow, and it is accurate */

            f64 f2s = sqrt(one + g2 / f2);
            /* Do the F2S(real)*FS(complex) multiply with two real
               multiplies */
            r = CMPLX(f2s * creal(fs), f2s * cimag(fs));
            cs = one / f2s;
            f64 d = f2 + g2;
            /* Do complex/real division explicitly with two real divisions */
            sn = CMPLX(creal(r) / d, cimag(r) / d);
            sn = sn * conj(gs);
            if (count != 0) {
                if (count > 0) {
                    for (INT j = 0; j < count; j++) {
                        r = r * safmx2;
                    }
                } else {
                    for (INT j = 0; j < -count; j++) {
                        r = r * safmn2;
                    }
                }
            }
        }

label50:
        C[ic] = cs;
        Y[iy] = sn;
        X[ix] = r;
        ic += incc;
        iy += incy;
        ix += incx;
    }
}
