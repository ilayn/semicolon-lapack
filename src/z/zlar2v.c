/**
 * @file zlar2v.c
 * @brief ZLAR2V applies a vector of complex plane rotations with real cosines
 *        and complex sines from both sides to a sequence of 2-by-2 Hermitian matrices.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLAR2V applies a vector of complex plane rotations with real cosines
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
void zlar2v(
    const int n,
    c128* const restrict X,
    c128* const restrict Y,
    c128* const restrict Z,
    const int incx,
    const f64* const restrict C,
    const c128* const restrict S,
    const int incc)
{
    int i, ic, ix;
    f64 ci, sii, sir, t1i, t1r, t5, t6, xi, yi, zii, zir;
    c128 si, t2, t3, t4, zi;

    ix = 0;
    ic = 0;
    for (i = 0; i < n; i++) {
        xi = creal(X[ix]);
        yi = creal(Y[ix]);
        zi = Z[ix];
        zir = creal(zi);
        zii = cimag(zi);
        ci = C[ic];
        si = S[ic];
        sir = creal(si);
        sii = cimag(si);
        t1r = sir * zir - sii * zii;
        t1i = sir * zii + sii * zir;
        t2 = ci * zi;
        t3 = t2 - conj(si) * xi;
        t4 = conj(t2) + si * yi;
        t5 = ci * xi + t1r;
        t6 = ci * yi - t1r;
        X[ix] = CMPLX(ci * t5 + (sir * creal(t4) + sii * cimag(t4)), 0.0);
        Y[ix] = CMPLX(ci * t6 - (sir * creal(t3) - sii * cimag(t3)), 0.0);
        Z[ix] = ci * t3 + conj(si) * CMPLX(t6, t1i);
        ix = ix + incx;
        ic = ic + incc;
    }
}
