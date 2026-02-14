/**
 * @file zspmv.c
 * @brief ZSPMV performs the matrix-vector operation y := alpha*A*x + beta*y for complex symmetric packed matrices.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSPMV performs the matrix-vector operation
 *
 *    y := alpha*A*x + beta*y,
 *
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n symmetric matrix, supplied in packed form.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the matrix A is supplied in the packed array AP:
 *                       = 'U': Upper triangular part of A is supplied in AP
 *                       = 'L': Lower triangular part of A is supplied in AP
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     alpha  The scalar alpha.
 * @param[in]     AP     The packed matrix A. Dimension at least (n*(n+1))/2.
 * @param[in]     X      The vector x. Dimension at least (1 + (n-1)*abs(incx)).
 * @param[in]     incx   The increment for elements of X. incx != 0.
 * @param[in]     beta   The scalar beta.
 * @param[in,out] Y      The vector y. Dimension at least (1 + (n-1)*abs(incy)).
 * @param[in]     incy   The increment for elements of Y. incy != 0.
 */
void zspmv(
    const char* uplo,
    const int n,
    const c128 alpha,
    const c128* const restrict AP,
    const c128* const restrict X,
    const int incx,
    const c128 beta,
    c128* const restrict Y,
    const int incy)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

    int i, info, ix, iy, j, jx, jy, k, kk, kx, ky;
    c128 temp1, temp2;

    info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (incx == 0) {
        info = 6;
    } else if (incy == 0) {
        info = 9;
    }
    if (info != 0) {
        xerbla("ZSPMV ", info);
        return;
    }

    if ((n == 0) || ((alpha == ZERO) && (beta == ONE)))
        return;

    if (incx > 0) {
        kx = 0;
    } else {
        kx = -(n - 1) * incx;
    }
    if (incy > 0) {
        ky = 0;
    } else {
        ky = -(n - 1) * incy;
    }

    if (beta != ONE) {
        if (incy == 1) {
            if (beta == ZERO) {
                for (i = 0; i < n; i++) {
                    Y[i] = ZERO;
                }
            } else {
                for (i = 0; i < n; i++) {
                    Y[i] = beta * Y[i];
                }
            }
        } else {
            iy = ky;
            if (beta == ZERO) {
                for (i = 0; i < n; i++) {
                    Y[iy] = ZERO;
                    iy = iy + incy;
                }
            } else {
                for (i = 0; i < n; i++) {
                    Y[iy] = beta * Y[iy];
                    iy = iy + incy;
                }
            }
        }
    }
    if (alpha == ZERO)
        return;
    kk = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {

        if ((incx == 1) && (incy == 1)) {
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[j];
                temp2 = ZERO;
                k = kk;
                for (i = 0; i < j; i++) {
                    Y[i] = Y[i] + temp1 * AP[k];
                    temp2 = temp2 + AP[k] * X[i];
                    k = k + 1;
                }
                Y[j] = Y[j] + temp1 * AP[kk + j] + alpha * temp2;
                kk = kk + j + 1;
            }
        } else {
            jx = kx;
            jy = ky;
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[jx];
                temp2 = ZERO;
                ix = kx;
                iy = ky;
                for (k = kk; k < kk + j; k++) {
                    Y[iy] = Y[iy] + temp1 * AP[k];
                    temp2 = temp2 + AP[k] * X[ix];
                    ix = ix + incx;
                    iy = iy + incy;
                }
                Y[jy] = Y[jy] + temp1 * AP[kk + j] + alpha * temp2;
                jx = jx + incx;
                jy = jy + incy;
                kk = kk + j + 1;
            }
        }
    } else {

        if ((incx == 1) && (incy == 1)) {
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[j];
                temp2 = ZERO;
                Y[j] = Y[j] + temp1 * AP[kk];
                k = kk + 1;
                for (i = j + 1; i < n; i++) {
                    Y[i] = Y[i] + temp1 * AP[k];
                    temp2 = temp2 + AP[k] * X[i];
                    k = k + 1;
                }
                Y[j] = Y[j] + alpha * temp2;
                kk = kk + (n - j);
            }
        } else {
            jx = kx;
            jy = ky;
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[jx];
                temp2 = ZERO;
                Y[jy] = Y[jy] + temp1 * AP[kk];
                ix = jx;
                iy = jy;
                for (k = kk + 1; k < kk + n - j; k++) {
                    ix = ix + incx;
                    iy = iy + incy;
                    Y[iy] = Y[iy] + temp1 * AP[k];
                    temp2 = temp2 + AP[k] * X[ix];
                }
                Y[jy] = Y[jy] + alpha * temp2;
                jx = jx + incx;
                jy = jy + incy;
                kk = kk + (n - j);
            }
        }
    }
}
