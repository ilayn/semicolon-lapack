/**
 * @file clar2v.c
 * @brief CLAR2V applies a vector of complex plane rotations with real cosines
 *        and complex sines from both sides to a sequence of 2-by-2 Hermitian matrices.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLAR2V applies a vector of complex plane rotations with real cosines
 * from both sides to a sequence of 2-by-2 complex Hermitian matrices,
 * defined by the elements of the vectors x, y and z. For i = 1,2,...,n
 *
 *    (       x(i)  z(i) ) :=
 *    ( conjg(z(i)) y(i) )
 *
 *      (  c(i) conjg(s(i)) ) (       x(i)  z(i) ) ( c(i) -conjg(s(i)) )
 *      ( -s(i)       c(i)  ) ( conjg(z(i)) y(i) ) ( s(i)        c(i)  )
 *
 * @param[in]     n      The number of plane rotations to be applied.
 * @param[in,out] X      The vector x; the elements of x are assumed to be real.
 *                        Array of dimension (1+(n-1)*incx).
 * @param[in,out] Y      The vector y; the elements of y are assumed to be real.
 *                        Array of dimension (1+(n-1)*incx).
 * @param[in,out] Z      The vector z. Array of dimension (1+(n-1)*incx).
 * @param[in]     incx   The increment between elements of X, Y and Z. incx > 0.
 * @param[in]     C      The cosines of the plane rotations.
 *                        Array of dimension (1+(n-1)*incc).
 * @param[in]     S      The sines of the plane rotations.
 *                        Array of dimension (1+(n-1)*incc).
 * @param[in]     incc   The increment between elements of C and S. incc > 0.
 */
void clar2v(
    const INT n,
    c64* restrict X,
    c64* restrict Y,
    c64* restrict Z,
    const INT incx,
    const f32* restrict C,
    const c64* restrict S,
    const INT incc)
{
    INT i, ic, ix;
    f32 ci, sii, sir, t1i, t1r, t5, t6, xi, yi, zii, zir;
    c64 si, t2, t3, t4, zi;

    ix = 0;
    ic = 0;
    for (i = 0; i < n; i++) {
        xi = crealf(X[ix]);
        yi = crealf(Y[ix]);
        zi = Z[ix];
        zir = crealf(zi);
        zii = cimagf(zi);
        ci = C[ic];
        si = S[ic];
        sir = crealf(si);
        sii = cimagf(si);
        t1r = sir * zir - sii * zii;
        t1i = sir * zii + sii * zir;
        t2 = ci * zi;
        t3 = t2 - conjf(si) * xi;
        t4 = conjf(t2) + si * yi;
        t5 = ci * xi + t1r;
        t6 = ci * yi - t1r;
        X[ix] = CMPLXF(ci * t5 + (sir * crealf(t4) + sii * cimagf(t4)), 0.0f);
        Y[ix] = CMPLXF(ci * t6 - (sir * crealf(t3) - sii * cimagf(t3)), 0.0f);
        Z[ix] = ci * t3 + conjf(si) * CMPLXF(t6, t1i);
        ix = ix + incx;
        ic = ic + incc;
    }
}
