/**
 * @file zsbmv.c
 * @brief ZSBMV performs the matrix-vector operation y := alpha*A*x + beta*y,
 *        where A is a complex symmetric band matrix.
 */

#include "verify.h"

/**
 * ZSBMV performs the matrix-vector operation
 *
 *    y := alpha*A*x + beta*y,
 *
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n symmetric band matrix, with k super-diagonals.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the band matrix A is being supplied.
 *                        = 'U':  Upper triangular part of A is supplied.
 *                        = 'L':  Lower triangular part of A is supplied.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     k       The number of super-diagonals of A. k >= 0.
 * @param[in]     alpha   The scalar alpha.
 * @param[in]     A       The band matrix, dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= k + 1.
 * @param[in]     X       The vector x, dimension at least (1 + (n-1)*abs(incx)).
 * @param[in]     incx    The increment for elements of X. incx != 0.
 * @param[in]     beta    The scalar beta.
 * @param[in,out] Y       The vector y, dimension at least (1 + (n-1)*abs(incy)).
 * @param[in]     incy    The increment for elements of Y. incy != 0.
 */
void zsbmv(const char* uplo, const INT n, const INT k,
           const c128 alpha, const c128* A, const INT lda,
           const c128* X, const INT incx,
           const c128 beta, c128* Y, const INT incy)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT i, ix, iy, j, jx, jy, kx, ky, l;
    INT istart, iend;
    c128 temp1, temp2;

    /* Test the input parameters. */

    if ((uplo[0] != 'U' && uplo[0] != 'u' && uplo[0] != 'L' && uplo[0] != 'l') ||
        n < 0 || k < 0 || lda < k + 1 || incx == 0 || incy == 0) {
        return;
    }

    /* Quick return if possible. */

    if (n == 0 || (alpha == CZERO && beta == CONE))
        return;

    /* Set up the start points in X and Y. */

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

    /* First form y := beta*y. */

    if (beta != CONE) {
        if (incy == 1) {
            if (beta == CZERO) {
                for (i = 0; i < n; i++) {
                    Y[i] = CZERO;
                }
            } else {
                for (i = 0; i < n; i++) {
                    Y[i] = beta * Y[i];
                }
            }
        } else {
            iy = ky;
            if (beta == CZERO) {
                for (i = 0; i < n; i++) {
                    Y[iy] = CZERO;
                    iy += incy;
                }
            } else {
                for (i = 0; i < n; i++) {
                    Y[iy] = beta * Y[iy];
                    iy += incy;
                }
            }
        }
    }
    if (alpha == CZERO)
        return;
    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /* Form y when upper triangle of A is stored. */

        if (incx == 1 && incy == 1) {
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[j];
                temp2 = CZERO;
                l = k - j;
                istart = (0 > j - k) ? 0 : j - k;
                for (i = istart; i < j; i++) {
                    Y[i] += temp1 * A[l + i + j * lda];
                    temp2 += A[l + i + j * lda] * X[i];
                }
                Y[j] += temp1 * A[k + j * lda] + alpha * temp2;
            }
        } else {
            jx = kx;
            jy = ky;
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[jx];
                temp2 = CZERO;
                ix = kx;
                iy = ky;
                l = k - j;
                istart = (0 > j - k) ? 0 : j - k;
                for (i = istart; i < j; i++) {
                    Y[iy] += temp1 * A[l + i + j * lda];
                    temp2 += A[l + i + j * lda] * X[ix];
                    ix += incx;
                    iy += incy;
                }
                Y[jy] += temp1 * A[k + j * lda] + alpha * temp2;
                jx += incx;
                jy += incy;
                if (j >= k) {
                    kx += incx;
                    ky += incy;
                }
            }
        }
    } else {

        /* Form y when lower triangle of A is stored. */

        if (incx == 1 && incy == 1) {
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[j];
                temp2 = CZERO;
                Y[j] += temp1 * A[j * lda];
                l = -j;
                iend = (n < j + k + 1) ? n : j + k + 1;
                for (i = j + 1; i < iend; i++) {
                    Y[i] += temp1 * A[l + i + j * lda];
                    temp2 += A[l + i + j * lda] * X[i];
                }
                Y[j] += alpha * temp2;
            }
        } else {
            jx = kx;
            jy = ky;
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[jx];
                temp2 = CZERO;
                Y[jy] += temp1 * A[j * lda];
                l = -j;
                ix = jx;
                iy = jy;
                iend = (n < j + k + 1) ? n : j + k + 1;
                for (i = j + 1; i < iend; i++) {
                    ix += incx;
                    iy += incy;
                    Y[iy] += temp1 * A[l + i + j * lda];
                    temp2 += A[l + i + j * lda] * X[ix];
                }
                Y[jy] += alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    }
}
