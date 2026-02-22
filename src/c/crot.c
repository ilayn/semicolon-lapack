/**
 * @file crot.c
 * @brief CROT applies a plane rotation with real cosine and complex sine
 *        to a pair of complex vectors.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CROT applies a plane rotation, where the cos (C) is real and the
 * sin (S) is complex, and the vectors CX and CY are complex.
 *
 * @param[in]     n     The number of elements in the vectors CX and CY.
 * @param[in,out] CX    Complex*16 array, dimension (N).
 *                      On input, the vector X.
 *                      On output, CX is overwritten with C*X + S*Y.
 * @param[in]     incx  The increment between successive values of CX. INCX <> 0.
 * @param[in,out] CY    Complex*16 array, dimension (N).
 *                      On input, the vector Y.
 *                      On output, CY is overwritten with -CONJG(S)*X + C*Y.
 * @param[in]     incy  The increment between successive values of CY. INCY <> 0.
 * @param[in]     c     Single precision.
 * @param[in]     s     Complex*16.
 *                      C and S define a rotation
 *                         [  C          S  ]
 *                         [ -conjg(S)   C  ]
 *                      where C*C + S*CONJG(S) = 1.0.
 */
void crot(const INT n, c64* restrict CX, const INT incx,
          c64* restrict CY, const INT incy,
          const f32 c, const c64 s)
{
    INT i, ix, iy;
    c64 stemp;

    if (n <= 0) return;

    if (incx == 1 && incy == 1) {
        /* Code for both increments equal to 1 */
        for (i = 0; i < n; i++) {
            stemp = c * CX[i] + s * CY[i];
            CY[i] = c * CY[i] - conjf(s) * CX[i];
            CX[i] = stemp;
        }
    } else {
        /* Code for unequal increments or equal increments not equal to 1 */
        ix = 0;
        iy = 0;
        if (incx < 0) ix = (-n + 1) * incx;
        if (incy < 0) iy = (-n + 1) * incy;
        for (i = 0; i < n; i++) {
            stemp = c * CX[ix] + s * CY[iy];
            CY[iy] = c * CY[iy] - conjf(s) * CX[ix];
            CX[ix] = stemp;
            ix += incx;
            iy += incy;
        }
    }
}
