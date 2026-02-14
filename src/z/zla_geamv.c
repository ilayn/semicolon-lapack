/**
 * @file zla_geamv.c
 * @brief ZLA_GEAMV computes a matrix-vector product using a general matrix
 *        to calculate error bounds.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLA_GEAMV performs one of the matrix-vector operations
 *
 *    y := alpha*abs(A)*abs(x) + beta*abs(y),
 * or y := alpha*abs(A)**T*abs(x) + beta*abs(y),
 *
 * where alpha and beta are scalars, x and y are vectors and A is an
 * m by n matrix.
 *
 * This function is primarily used in calculating error bounds.
 * To protect against underflow during evaluation, components in
 * the resulting vector are perturbed away from zero by (N+1)
 * times the underflow threshold.  To prevent unnecessarily large
 * errors for block-structure embedded in general matrices,
 * "symbolically" zero components are not perturbed.  A zero
 * entry is considered "symbolic" if all multiplications involved
 * in computing that entry have at least one zero multiplicand.
 *
 * @param[in]     trans  Specifies the operation to be performed:
 *                       CblasNoTrans:   y := alpha*abs(A)*abs(x) + beta*abs(y)
 *                       CblasTrans:     y := alpha*abs(A**T)*abs(x) + beta*abs(y)
 *                       CblasConjTrans: y := alpha*abs(A**T)*abs(x) + beta*abs(y)
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= 0.
 * @param[in]     alpha  The scalar alpha.
 * @param[in]     A      Complex*16 array, dimension (lda, n).
 *                       The leading m by n part of the array A must
 *                       contain the matrix of coefficients.
 * @param[in]     lda    The first dimension of A. lda >= max(1, m).
 * @param[in]     X      Complex*16 array, dimension at least
 *                       (1 + (n-1)*abs(incx)) when trans = CblasNoTrans
 *                       and at least (1 + (m-1)*abs(incx)) otherwise.
 * @param[in]     incx   The increment for the elements of X. incx != 0.
 * @param[in]     beta   The scalar beta.
 * @param[in,out] Y      Double precision array, dimension
 *                       (1 + (m-1)*abs(incy)) when trans = CblasNoTrans
 *                       and at least (1 + (n-1)*abs(incy)) otherwise.
 * @param[in]     incy   The increment for the elements of Y. incy != 0.
 */
void zla_geamv(const int trans, const int m, const int n,
               const f64 alpha, const c128* const restrict A,
               const int lda, const c128* const restrict X,
               const int incx, const f64 beta,
               f64* const restrict Y, const int incy)
{
    const c128 CZERO = CMPLX(0.0, 0.0);

    int symb_zero;
    f64 temp, safe1;
    int i, info, iy, j, jx, kx, ky, lenx, leny;

    info = 0;
    if (!(trans == CblasNoTrans || trans == CblasTrans || trans == CblasConjTrans)) {
        info = 1;
    } else if (m < 0) {
        info = 2;
    } else if (n < 0) {
        info = 3;
    } else if (lda < ((1 > m) ? 1 : m)) {
        info = 6;
    } else if (incx == 0) {
        info = 8;
    } else if (incy == 0) {
        info = 11;
    }
    if (info != 0) {
        xerbla("ZLA_GEAMV ", info);
        return;
    }

    if ((m == 0) || (n == 0) ||
        ((alpha == 0.0) && (beta == 1.0))) {
        return;
    }

    if (trans == CblasNoTrans) {
        lenx = n;
        leny = m;
    } else {
        lenx = m;
        leny = n;
    }
    if (incx > 0) {
        kx = 0;
    } else {
        kx = -(lenx - 1) * incx;
    }
    if (incy > 0) {
        ky = 0;
    } else {
        ky = -(leny - 1) * incy;
    }

    safe1 = dlamch("Safe minimum");
    safe1 = (n + 1) * safe1;

    iy = ky;
    if (incx == 1) {
        if (trans == CblasNoTrans) {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0) {
                    symb_zero = 1;
                    Y[iy] = 0.0;
                } else if (Y[iy] == 0.0) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabs(Y[iy]);
                }
                if (alpha != 0.0) {
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1(A[i + j * lda]);
                        symb_zero = symb_zero &&
                            (X[j] == CZERO || temp == 0.0);

                        Y[iy] = Y[iy] + alpha * cabs1(X[j]) * temp;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysign(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        } else {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0) {
                    symb_zero = 1;
                    Y[iy] = 0.0;
                } else if (Y[iy] == 0.0) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabs(Y[iy]);
                }
                if (alpha != 0.0) {
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1(A[j + i * lda]);
                        symb_zero = symb_zero &&
                            (X[j] == CZERO || temp == 0.0);

                        Y[iy] = Y[iy] + alpha * cabs1(X[j]) * temp;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysign(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        }
    } else {
        if (trans == CblasNoTrans) {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0) {
                    symb_zero = 1;
                    Y[iy] = 0.0;
                } else if (Y[iy] == 0.0) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabs(Y[iy]);
                }
                if (alpha != 0.0) {
                    jx = kx;
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1(A[i + j * lda]);
                        symb_zero = symb_zero &&
                            (X[jx] == CZERO || temp == 0.0);

                        Y[iy] = Y[iy] + alpha * cabs1(X[jx]) * temp;
                        jx = jx + incx;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysign(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        } else {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0) {
                    symb_zero = 1;
                    Y[iy] = 0.0;
                } else if (Y[iy] == 0.0) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabs(Y[iy]);
                }
                if (alpha != 0.0) {
                    jx = kx;
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1(A[j + i * lda]);
                        symb_zero = symb_zero &&
                            (X[jx] == CZERO || temp == 0.0);

                        Y[iy] = Y[iy] + alpha * cabs1(X[jx]) * temp;
                        jx = jx + incx;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysign(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        }
    }
}
