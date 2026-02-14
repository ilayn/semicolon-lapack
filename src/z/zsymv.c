/**
 * @file zsymv.c
 * @brief ZSYMV computes a matrix-vector product for a complex symmetric matrix.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSYMV performs the matrix-vector operation
 *
 *    y := alpha*A*x + beta*y,
 *
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n symmetric matrix.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the array A is to be referenced:
 *                       = 'U': Only the upper triangular part of A is referenced.
 *                       = 'L': Only the lower triangular part of A is referenced.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     alpha  Specifies the scalar alpha.
 * @param[in]     A      Double complex array, dimension (lda, n).
 *                       The symmetric matrix A.
 * @param[in]     lda    The first dimension of A. lda >= max(1, n).
 * @param[in]     X      Double complex array, dimension at least
 *                       (1 + (n-1)*abs(incx)).
 *                       The n-element vector x.
 * @param[in]     incx   The increment for the elements of X. incx must not
 *                       be zero.
 * @param[in]     beta   Specifies the scalar beta.
 * @param[in,out] Y      Double complex array, dimension at least
 *                       (1 + (n-1)*abs(incy)).
 *                       On entry, the n-element vector y. On exit, the
 *                       updated vector y.
 * @param[in]     incy   The increment for the elements of Y. incy must not
 *                       be zero.
 */
void zsymv(const char* uplo, const int n,
           const double complex alpha,
           const double complex* const restrict A, const int lda,
           const double complex* const restrict X, const int incx,
           const double complex beta,
           double complex* const restrict Y, const int incy)
{
    const double complex ONE = CMPLX(1.0, 0.0);
    const double complex ZERO = CMPLX(0.0, 0.0);

    int i, info, ix, iy, j, jx, jy, kx, ky;
    double complex temp1, temp2;

    /* Test the input parameters. */
    info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (lda < (1 > n ? 1 : n)) {
        info = 5;
    } else if (incx == 0) {
        info = 7;
    } else if (incy == 0) {
        info = 10;
    }
    if (info != 0) {
        xerbla("ZSYMV ", info);
        return;
    }

    /* Quick return if possible. */
    if ((n == 0) || (alpha == ZERO && beta == ONE)) {
        return;
    }

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

    /*
     * Start the operations. In this version the elements of A are
     * accessed sequentially with one pass through the triangular part
     * of A.
     *
     * First form  y := beta*y.
     */
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
    if (alpha == ZERO) {
        return;
    }
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Form  y  when A is stored in upper triangle. */
        if ((incx == 1) && (incy == 1)) {
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[j];
                temp2 = ZERO;
                for (i = 0; i < j; i++) {
                    Y[i] = Y[i] + temp1 * A[i + j * lda];
                    temp2 = temp2 + A[i + j * lda] * X[i];
                }
                Y[j] = Y[j] + temp1 * A[j + j * lda] + alpha * temp2;
            }
        } else {
            jx = kx;
            jy = ky;
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[jx];
                temp2 = ZERO;
                ix = kx;
                iy = ky;
                for (i = 0; i < j; i++) {
                    Y[iy] = Y[iy] + temp1 * A[i + j * lda];
                    temp2 = temp2 + A[i + j * lda] * X[ix];
                    ix += incx;
                    iy += incy;
                }
                Y[jy] = Y[jy] + temp1 * A[j + j * lda] + alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    } else {
        /* Form  y  when A is stored in lower triangle. */
        if ((incx == 1) && (incy == 1)) {
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[j];
                temp2 = ZERO;
                Y[j] = Y[j] + temp1 * A[j + j * lda];
                for (i = j + 1; i < n; i++) {
                    Y[i] = Y[i] + temp1 * A[i + j * lda];
                    temp2 = temp2 + A[i + j * lda] * X[i];
                }
                Y[j] = Y[j] + alpha * temp2;
            }
        } else {
            jx = kx;
            jy = ky;
            for (j = 0; j < n; j++) {
                temp1 = alpha * X[jx];
                temp2 = ZERO;
                Y[jy] = Y[jy] + temp1 * A[j + j * lda];
                ix = jx;
                iy = jy;
                for (i = j + 1; i < n; i++) {
                    ix += incx;
                    iy += incy;
                    Y[iy] = Y[iy] + temp1 * A[i + j * lda];
                    temp2 = temp2 + A[i + j * lda] * X[ix];
                }
                Y[jy] = Y[jy] + alpha * temp2;
                jx += incx;
                jy += incy;
            }
        }
    }

    return;
}
