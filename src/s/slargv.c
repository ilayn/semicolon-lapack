/**
 * @file slargv.c
 * @brief SLARGV generates a vector of plane rotations with real cosines and real sines.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLARGV generates a vector of real plane rotations, determined by
 * elements of the real vectors x and y. For i = 0,1,...,n-1
 *
 *    (  c(i)  s(i) ) ( x(i) ) = ( r(i) )
 *    ( -s(i)  c(i) ) ( y(i) ) = (   0  )
 *
 * where c(i)**2 + s(i)**2 = 1.
 *
 * @param[in]     n     The number of plane rotations to be generated.
 * @param[in,out] X     Double precision array, dimension (1+(n-1)*incx).
 *                      On entry, the vector x.
 *                      On exit, x(i) is overwritten by r(i), for i = 0,...,n-1.
 * @param[in]     incx  The increment between elements of X. incx > 0.
 * @param[in,out] Y     Double precision array, dimension (1+(n-1)*incy).
 *                      On entry, the vector y.
 *                      On exit, the sines of the plane rotations.
 * @param[in]     incy  The increment between elements of Y. incy > 0.
 * @param[out]    C     Double precision array, dimension (1+(n-1)*incc).
 *                      The cosines of the plane rotations.
 * @param[in]     incc  The increment between elements of C. incc > 0.
 */
void slargv(const INT n, f32* restrict X, const INT incx,
            f32* restrict Y, const INT incy,
            f32* restrict C, const INT incc)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    INT ix = 0;
    INT iy = 0;
    INT ic = 0;

    for (INT i = 0; i < n; i++) {
        f32 f = X[ix];
        f32 g = Y[iy];

        if (g == zero) {
            /* g is zero, no rotation needed */
            C[ic] = one;
        } else if (f == zero) {
            /* f is zero, swap and set rotation */
            C[ic] = zero;
            Y[iy] = one;
            X[ix] = g;
        } else if (fabsf(f) > fabsf(g)) {
            /* |f| > |g|: standard case */
            f32 t = g / f;
            f32 tt = sqrtf(one + t * t);
            C[ic] = one / tt;
            Y[iy] = t * C[ic];
            X[ix] = f * tt;
        } else {
            /* |f| <= |g|: reversed case */
            f32 t = f / g;
            f32 tt = sqrtf(one + t * t);
            Y[iy] = one / tt;
            C[ic] = t * Y[iy];
            X[ix] = g * tt;
        }

        ic += incc;
        iy += incy;
        ix += incx;
    }
}
