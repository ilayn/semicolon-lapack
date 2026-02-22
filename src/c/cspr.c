/**
 * @file cspr.c
 * @brief CSPR performs the symmetrical rank-1 update of a complex symmetric packed matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CSPR performs the symmetric rank 1 operation
 *
 *    A := alpha*x*x**T + A,
 *
 * where alpha is a complex scalar, x is an n element vector and A is an
 * n by n symmetric matrix, supplied in packed form.
 *
 * @param[in]     uplo   On entry, UPLO specifies whether the upper or lower
 *                       triangular part of the matrix A is supplied in the packed
 *                       array AP as follows:
 *                         - = 'U' or 'u': The upper triangular part of A is
 *                                         supplied in AP.
 *                         - = 'L' or 'l': The lower triangular part of A is
 *                                         supplied in AP.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     alpha  The scalar alpha.
 * @param[in]     X      Complex*16 array, dimension at least (1 + (n-1)*abs(incx)).
 *                       The n-element vector x.
 * @param[in]     incx   The increment for the elements of X. incx must not be zero.
 * @param[in,out] AP     Complex*16 array, dimension at least (n*(n+1)/2).
 *                       On entry, the upper or lower triangular part of the
 *                       symmetric matrix packed sequentially, column by column.
 *                       On exit, overwritten by the updated matrix.
 */
void cspr(
    const char* uplo,
    const INT n,
    const c64 alpha,
    const c64* restrict X,
    const INT incx,
    c64* restrict AP)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    INT i, info, ix, j, jx, k, kk, kx;
    c64 temp;

    info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (incx == 0) {
        info = 5;
    }
    if (info != 0) {
        xerbla("CSPR  ", info);
        return;
    }

    if ((n == 0) || (alpha == ZERO))
        return;

    if (incx <= 0) {
        kx = -(n - 1) * incx;
    } else if (incx != 1) {
        kx = 0;
    }

    kk = 0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {

        if (incx == 1) {
            for (j = 0; j < n; j++) {
                if (X[j] != ZERO) {
                    temp = alpha * X[j];
                    k = kk;
                    for (i = 0; i < j; i++) {
                        AP[k] = AP[k] + X[i] * temp;
                        k = k + 1;
                    }
                    AP[kk + j] = AP[kk + j] + X[j] * temp;
                } else {
                    AP[kk + j] = AP[kk + j];
                }
                kk = kk + j + 1;
            }
        } else {
            jx = kx;
            for (j = 0; j < n; j++) {
                if (X[jx] != ZERO) {
                    temp = alpha * X[jx];
                    ix = kx;
                    for (k = kk; k < kk + j; k++) {
                        AP[k] = AP[k] + X[ix] * temp;
                        ix = ix + incx;
                    }
                    AP[kk + j] = AP[kk + j] + X[jx] * temp;
                } else {
                    AP[kk + j] = AP[kk + j];
                }
                jx = jx + incx;
                kk = kk + j + 1;
            }
        }
    } else {

        if (incx == 1) {
            for (j = 0; j < n; j++) {
                if (X[j] != ZERO) {
                    temp = alpha * X[j];
                    AP[kk] = AP[kk] + temp * X[j];
                    k = kk + 1;
                    for (i = j + 1; i < n; i++) {
                        AP[k] = AP[k] + X[i] * temp;
                        k = k + 1;
                    }
                } else {
                    AP[kk] = AP[kk];
                }
                kk = kk + n - j;
            }
        } else {
            jx = kx;
            for (j = 0; j < n; j++) {
                if (X[jx] != ZERO) {
                    temp = alpha * X[jx];
                    AP[kk] = AP[kk] + temp * X[jx];
                    ix = jx;
                    for (k = kk + 1; k < kk + n - j; k++) {
                        ix = ix + incx;
                        AP[k] = AP[k] + X[ix] * temp;
                    }
                } else {
                    AP[kk] = AP[kk];
                }
                jx = jx + incx;
                kk = kk + n - j;
            }
        }
    }
}
