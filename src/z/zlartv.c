/**
 * @file zlartv.c
 * @brief ZLARTV applies a vector of plane rotations with real cosines and complex sines.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLARTV applies a vector of complex plane rotations with real cosines
 * to elements of the complex vectors x and y. For i = 0,1,...,n-1
 *
 *    ( x(i) ) := (        c(i)   s(i) ) ( x(i) )
 *    ( y(i) )    ( -conjg(s(i))  c(i) ) ( y(i) )
 *
 * @param[in]     n     The number of plane rotations to be applied.
 * @param[in,out] X     Complex*16 array, dimension (1+(n-1)*incx).
 *                      The vector x.
 * @param[in]     incx  The increment between elements of X. incx > 0.
 * @param[in,out] Y     Complex*16 array, dimension (1+(n-1)*incy).
 *                      The vector y.
 * @param[in]     incy  The increment between elements of Y. incy > 0.
 * @param[in]     C     Double precision array, dimension (1+(n-1)*incc).
 *                      The cosines of the plane rotations.
 * @param[in]     S     Complex*16 array, dimension (1+(n-1)*incc).
 *                      The sines of the plane rotations.
 * @param[in]     incc  The increment between elements of C and S. incc > 0.
 */
void zlartv(const INT n, c128* restrict X, const INT incx,
            c128* restrict Y, const INT incy,
            const f64* restrict C, const c128* restrict S,
            const INT incc)
{
    INT ix = 0;
    INT iy = 0;
    INT ic = 0;

    for (INT i = 0; i < n; i++) {
        c128 xi = X[ix];
        c128 yi = Y[iy];
        X[ix] = C[ic] * xi + S[ic] * yi;
        Y[iy] = C[ic] * yi - conj(S[ic]) * xi;
        ix += incx;
        iy += incy;
        ic += incc;
    }
}
