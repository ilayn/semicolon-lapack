/**
 * @file dlapll.c
 * @brief DLAPLL measures the linear dependence of two vectors.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * Given two column vectors X and Y, let
 *
 *                      A = ( X Y ).
 *
 * The subroutine first computes the QR factorization of A = Q*R,
 * and then computes the SVD of the 2-by-2 upper triangular matrix R.
 * The smaller singular value of R is returned in SSMIN, which is used
 * as the measurement of the linear dependency of the vectors X and Y.
 *
 * @param[in]     n      The length of the vectors X and Y.
 * @param[in,out] x      On entry, X contains the N-vector X.
 *                       On exit, X is overwritten.
 * @param[in]     incx   The increment between successive elements of X. INCX > 0.
 * @param[in,out] y      On entry, Y contains the N-vector Y.
 *                       On exit, Y is overwritten.
 * @param[in]     incy   The increment between successive elements of Y. INCY > 0.
 * @param[out]    ssmin  The smallest singular value of the N-by-2 matrix A = ( X Y ).
 */
void dlapll(const int n, f64* restrict x, const int incx,
            f64* restrict y, const int incy, f64* ssmin)
{
    f64 a11, a12, a22, c, tau, ssmax;

    if (n <= 1) {
        *ssmin = 0.0;
        return;
    }

    dlarfg(n, &x[0], &x[incx], incx, &tau);
    a11 = x[0];
    x[0] = 1.0;

    c = -tau * cblas_ddot(n, x, incx, y, incy);
    cblas_daxpy(n, c, x, incx, y, incy);

    dlarfg(n - 1, &y[incy], &y[2 * incy], incy, &tau);

    a12 = y[0];
    a22 = y[incy];

    dlas2(a11, a12, a22, ssmin, &ssmax);
}
