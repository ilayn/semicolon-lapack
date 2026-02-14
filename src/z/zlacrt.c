/**
 * @file zlacrt.c
 * @brief ZLACRT performs a linear transformation of a pair of complex vectors.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLACRT performs the operation
 *
 *    (  c  s )( x )  ==> ( x )
 *    ( -s  c )( y )      ( y )
 *
 * where c and s are complex and the vectors x and y are complex.
 *
 * @param[in]     n     The number of elements in the vectors CX and CY.
 * @param[in,out] cx    Complex array, dimension (N).
 *                      On input, the vector x.
 *                      On output, CX is overwritten with c*x + s*y.
 * @param[in]     incx  The increment between successive values of CX. incx <> 0.
 * @param[in,out] cy    Complex array, dimension (N).
 *                      On input, the vector y.
 *                      On output, CY is overwritten with -s*x + c*y.
 * @param[in]     incy  The increment between successive values of CY. incy <> 0.
 * @param[in]     c     Complex scalar.
 * @param[in]     s     Complex scalar.
 *                      C and S define the matrix
 *                         [  C   S  ].
 *                         [ -S   C  ]
 */
void zlacrt(
    const int n,
    double complex* const restrict cx,
    const int incx,
    double complex* const restrict cy,
    const int incy,
    const double complex c,
    const double complex s)
{
    int i, ix, iy;
    double complex ctemp;

    if (n <= 0) {
        return;
    }

    if (incx == 1 && incy == 1) {
        // Code for both increments equal to 1
        for (i = 0; i < n; i++) {
            ctemp = c * cx[i] + s * cy[i];
            cy[i] = c * cy[i] - s * cx[i];
            cx[i] = ctemp;
        }
        return;
    }

    // Code for unequal increments or equal increments not equal to 1
    ix = 0;
    iy = 0;
    if (incx < 0) {
        ix = (-n + 1) * incx;
    }
    if (incy < 0) {
        iy = (-n + 1) * incy;
    }
    for (i = 0; i < n; i++) {
        ctemp = c * cx[ix] + s * cy[iy];
        cy[iy] = c * cy[iy] - s * cx[ix];
        cx[ix] = ctemp;
        ix += incx;
        iy += incy;
    }
}
