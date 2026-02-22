/**
 * @file slar2v.c
 * @brief SLAR2V applies a vector of plane rotations from both sides to 2-by-2 symmetric matrices.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"

/**
 * SLAR2V applies a vector of real plane rotations from both sides to
 * a sequence of 2-by-2 real symmetric matrices, defined by the elements
 * of the vectors x, y and z.
 *
 * @param[in]     n      The number of plane rotations to be applied.
 * @param[in,out] X      The vector x. Array of dimension (1+(n-1)*incx).
 * @param[in,out] Y      The vector y. Array of dimension (1+(n-1)*incx).
 * @param[in,out] Z      The vector z. Array of dimension (1+(n-1)*incx).
 * @param[in]     incx   The increment between elements of X, Y and Z.
 * @param[in]     C      The cosines of the plane rotations. Array of dimension (1+(n-1)*incc).
 * @param[in]     S      The sines of the plane rotations. Array of dimension (1+(n-1)*incc).
 * @param[in]     incc   The increment between elements of C and S.
 */
void slar2v(
    const INT n,
    f32* restrict X,
    f32* restrict Y,
    f32* restrict Z,
    const INT incx,
    const f32* restrict C,
    const f32* restrict S,
    const INT incc)
{
    INT i, ic, ix;
    f32 ci, si, t1, t2, t3, t4, t5, t6, xi, yi, zi;

    ix = 0;
    ic = 0;
    for (i = 0; i < n; i++) {
        xi = X[ix];
        yi = Y[ix];
        zi = Z[ix];
        ci = C[ic];
        si = S[ic];
        t1 = si * zi;
        t2 = ci * zi;
        t3 = t2 - si * xi;
        t4 = t2 + si * yi;
        t5 = ci * xi + t1;
        t6 = ci * yi - t1;
        X[ix] = ci * t5 + si * t4;
        Y[ix] = ci * t6 - si * t3;
        Z[ix] = ci * t4 - si * t5;
        ix = ix + incx;
        ic = ic + incc;
    }
}
