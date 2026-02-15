/**
 * @file clargv.c
 * @brief CLARGV generates a vector of plane rotations with real cosines and complex sines.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

static inline f32 abs1(c64 ff) {
    f32 a = crealf(ff);
    f32 b = cimagf(ff);
    return (fabsf(a) > fabsf(b)) ? fabsf(a) : fabsf(b);
}

static inline f32 abssq(c64 ff) {
    return crealf(ff) * crealf(ff) + cimagf(ff) * cimagf(ff);
}

/**
 * CLARGV generates a vector of complex plane rotations with real
 * cosines, determined by elements of the complex vectors x and y.
 * For i = 0,1,...,n-1
 *
 *    (        c(i)   s(i) ) ( x(i) ) = ( r(i) )
 *    ( -conjg(s(i))  c(i) ) ( y(i) ) = (   0  )
 *
 *    where c(i)**2 + ABS(s(i))**2 = 1
 *
 * The following conventions are used (these are the same as in CLARTG,
 * but differ from the BLAS1 routine ZROTG):
 *    If y(i)=0, then c(i)=1 and s(i)=0.
 *    If x(i)=0, then c(i)=0 and s(i) is chosen so that r(i) is real.
 *
 * @param[in]     n     The number of plane rotations to be generated.
 * @param[in,out] X     Single complex array, dimension (1+(n-1)*incx).
 *                      On entry, the vector x.
 *                      On exit, x(i) is overwritten by r(i), for i = 0,...,n-1.
 * @param[in]     incx  The increment between elements of X. incx > 0.
 * @param[in,out] Y     Single complex array, dimension (1+(n-1)*incy).
 *                      On entry, the vector y.
 *                      On exit, the sines of the plane rotations.
 * @param[in]     incy  The increment between elements of Y. incy > 0.
 * @param[out]    C     Single precision array, dimension (1+(n-1)*incc).
 *                      The cosines of the plane rotations.
 * @param[in]     incc  The increment between elements of C. incc > 0.
 */
void clargv(const int n, c64* restrict X, const int incx,
            c64* restrict Y, const int incy,
            f32* restrict C, const int incc)
{
    const f32 two = 2.0f;
    const f32 one = 1.0f;
    const f32 zero = 0.0f;
    const c64 czero = CMPLXF(0.0f, 0.0f);

    f32 safmin = slamch("S");
    f32 eps = slamch("E");
    f32 safmn2 = powf(slamch("B"),
                        (int)(logf(safmin / eps) / logf(slamch("B")) / two));
    f32 safmx2 = one / safmn2;

    int ix = 0;
    int iy = 0;
    int ic = 0;

    for (int i = 0; i < n; i++) {
        c64 f = X[ix];
        c64 g = Y[iy];

        f32 scale = (abs1(f) > abs1(g)) ? abs1(f) : abs1(g);
        c64 fs = f;
        c64 gs = g;
        int count = 0;
        f32 cs;
        c64 sn, r;
        f32 f2, g2;

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
                r = slapy2(crealf(g), cimagf(g));
                /* Do complex/real division explicitly with two real
                   divisions */
                f32 d = slapy2(crealf(gs), cimagf(gs));
                sn = CMPLXF(crealf(gs) / d, -cimagf(gs) / d);
                goto label50;
            }
            f32 f2s = slapy2(crealf(fs), cimagf(fs));
            /* G2 and G2S are accurate
               G2 is at least SAFMIN, and G2S is at least SAFMN2 */
            f32 g2s = sqrtf(g2);
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
            c64 ff;
            f32 d;
            if (abs1(f) > one) {
                d = slapy2(crealf(f), cimagf(f));
                ff = CMPLXF(crealf(f) / d, cimagf(f) / d);
            } else {
                f32 dr = safmx2 * crealf(f);
                f32 di = safmx2 * cimagf(f);
                d = slapy2(dr, di);
                ff = CMPLXF(dr / d, di / d);
            }
            sn = ff * CMPLXF(crealf(gs) / g2s, -cimagf(gs) / g2s);
            r = cs * f + sn * g;
        } else {

            /* This is the most common case.
               Neither F2 nor F2/G2 are less than SAFMIN
               F2S cannot overflow, and it is accurate */

            f32 f2s = sqrtf(one + g2 / f2);
            /* Do the F2S(real)*FS(complex) multiply with two real
               multiplies */
            r = CMPLXF(f2s * crealf(fs), f2s * cimagf(fs));
            cs = one / f2s;
            f32 d = f2 + g2;
            /* Do complex/real division explicitly with two real divisions */
            sn = CMPLXF(crealf(r) / d, cimagf(r) / d);
            sn = sn * conjf(gs);
            if (count != 0) {
                if (count > 0) {
                    for (int j = 0; j < count; j++) {
                        r = r * safmx2;
                    }
                } else {
                    for (int j = 0; j < -count; j++) {
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
