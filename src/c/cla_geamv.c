/**
 * @file cla_geamv.c
 * @brief CLA_GEAMV computes a matrix-vector product using a general matrix
 *        to calculate error bounds.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CLA_GEAMV performs one of the matrix-vector operations
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
 * @param[in,out] Y      Single precision array, dimension
 *                       (1 + (m-1)*abs(incy)) when trans = CblasNoTrans
 *                       and at least (1 + (n-1)*abs(incy)) otherwise.
 * @param[in]     incy   The increment for the elements of Y. incy != 0.
 */
void cla_geamv(const INT trans, const INT m, const INT n,
               const f32 alpha, const c64* restrict A,
               const INT lda, const c64* restrict X,
               const INT incx, const f32 beta,
               f32* restrict Y, const INT incy)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    INT symb_zero;
    f32 temp, safe1;
    INT i, info, iy, j, jx, kx, ky, lenx, leny;

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
        xerbla("CLA_GEAMV ", info);
        return;
    }

    if ((m == 0) || (n == 0) ||
        ((alpha == 0.0f) && (beta == 1.0f))) {
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

    safe1 = slamch("Safe minimum");
    safe1 = (n + 1) * safe1;

    iy = ky;
    if (incx == 1) {
        if (trans == CblasNoTrans) {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0f) {
                    symb_zero = 1;
                    Y[iy] = 0.0f;
                } else if (Y[iy] == 0.0f) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabsf(Y[iy]);
                }
                if (alpha != 0.0f) {
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1f(A[i + j * lda]);
                        symb_zero = symb_zero &&
                            (X[j] == CZERO || temp == 0.0f);

                        Y[iy] = Y[iy] + alpha * cabs1f(X[j]) * temp;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysignf(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        } else {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0f) {
                    symb_zero = 1;
                    Y[iy] = 0.0f;
                } else if (Y[iy] == 0.0f) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabsf(Y[iy]);
                }
                if (alpha != 0.0f) {
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1f(A[j + i * lda]);
                        symb_zero = symb_zero &&
                            (X[j] == CZERO || temp == 0.0f);

                        Y[iy] = Y[iy] + alpha * cabs1f(X[j]) * temp;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysignf(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        }
    } else {
        if (trans == CblasNoTrans) {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0f) {
                    symb_zero = 1;
                    Y[iy] = 0.0f;
                } else if (Y[iy] == 0.0f) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabsf(Y[iy]);
                }
                if (alpha != 0.0f) {
                    jx = kx;
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1f(A[i + j * lda]);
                        symb_zero = symb_zero &&
                            (X[jx] == CZERO || temp == 0.0f);

                        Y[iy] = Y[iy] + alpha * cabs1f(X[jx]) * temp;
                        jx = jx + incx;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysignf(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        } else {
            for (i = 0; i < leny; i++) {
                if (beta == 0.0f) {
                    symb_zero = 1;
                    Y[iy] = 0.0f;
                } else if (Y[iy] == 0.0f) {
                    symb_zero = 1;
                } else {
                    symb_zero = 0;
                    Y[iy] = beta * fabsf(Y[iy]);
                }
                if (alpha != 0.0f) {
                    jx = kx;
                    for (j = 0; j < lenx; j++) {
                        temp = cabs1f(A[j + i * lda]);
                        symb_zero = symb_zero &&
                            (X[jx] == CZERO || temp == 0.0f);

                        Y[iy] = Y[iy] + alpha * cabs1f(X[jx]) * temp;
                        jx = jx + incx;
                    }
                }

                if (!symb_zero) {
                    Y[iy] = Y[iy] + copysignf(safe1, Y[iy]);
                }

                iy = iy + incy;
            }
        }
    }
}
