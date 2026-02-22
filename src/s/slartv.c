/**
 * @file slartv.c
 * @brief SLARTV applies a vector of plane rotations with real cosines and real sines.
 */

#include "semicolon_lapack_single.h"

/**
 * SLARTV applies a vector of real plane rotations to elements of the
 * real vectors x and y. For i = 0,1,...,n-1
 *
 *    ( x(i) ) := (  c(i)  s(i) ) ( x(i) )
 *    ( y(i) )    ( -s(i)  c(i) ) ( y(i) )
 *
 * @param[in]     n     The number of plane rotations to be applied.
 * @param[in,out] X     Double precision array, dimension (1+(n-1)*incx).
 *                      The vector x.
 * @param[in]     incx  The increment between elements of X. incx > 0.
 * @param[in,out] Y     Double precision array, dimension (1+(n-1)*incy).
 *                      The vector y.
 * @param[in]     incy  The increment between elements of Y. incy > 0.
 * @param[in]     C     Double precision array, dimension (1+(n-1)*incc).
 *                      The cosines of the plane rotations.
 * @param[in]     S     Double precision array, dimension (1+(n-1)*incc).
 *                      The sines of the plane rotations.
 * @param[in]     incc  The increment between elements of C and S. incc > 0.
 */
void slartv(const INT n, f32* restrict X, const INT incx,
            f32* restrict Y, const INT incy,
            const f32* restrict C, const f32* restrict S,
            const INT incc)
{
    INT ix = 0;
    INT iy = 0;
    INT ic = 0;

    for (INT i = 0; i < n; i++) {
        f32 xi = X[ix];
        f32 yi = Y[iy];
        X[ix] = C[ic] * xi + S[ic] * yi;
        Y[iy] = C[ic] * yi - S[ic] * xi;
        ix += incx;
        iy += incy;
        ic += incc;
    }
}
