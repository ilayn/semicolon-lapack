/**
 * @file csyr.c
 * @brief CSYR performs the symmetric rank-1 update of a complex symmetric matrix.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CSYR performs the symmetric rank 1 operation
 *
 *    A := alpha*x*x**H + A,
 *
 * where alpha is a complex scalar, x is an n element vector and A is an
 * n by n symmetric matrix.
 *
 * @param[in] uplo
 *          On entry, UPLO specifies whether the upper or lower
 *          triangular part of the array A is to be referenced as
 *          follows:
 *
 *             UPLO = 'U' or 'u'   Only the upper triangular part of A
 *                                 is to be referenced.
 *
 *             UPLO = 'L' or 'l'   Only the lower triangular part of A
 *                                 is to be referenced.
 *
 * @param[in] n
 *          On entry, N specifies the order of the matrix A.
 *          N must be at least zero.
 *
 * @param[in] alpha
 *          On entry, ALPHA specifies the scalar alpha.
 *
 * @param[in] X
 *          Single complex array, dimension at least
 *          ( 1 + ( N - 1 )*abs( INCX ) ).
 *          Before entry, the incremented array X must contain the N-
 *          element vector x.
 *
 * @param[in] incx
 *          On entry, INCX specifies the increment for the elements of
 *          X. INCX must not be zero.
 *
 * @param[in,out] A
 *          Single complex array, dimension ( lda, n ).
 *          Before entry, with  UPLO = 'U' or 'u', the leading n by n
 *          upper triangular part of the array A must contain the upper
 *          triangular part of the symmetric matrix and the strictly
 *          lower triangular part of A is not referenced. On exit, the
 *          upper triangular part of the array A is overwritten by the
 *          upper triangular part of the updated matrix.
 *          Before entry, with UPLO = 'L' or 'l', the leading n by n
 *          lower triangular part of the array A must contain the lower
 *          triangular part of the symmetric matrix and the strictly
 *          upper triangular part of A is not referenced. On exit, the
 *          lower triangular part of the array A is overwritten by the
 *          lower triangular part of the updated matrix.
 *
 * @param[in] lda
 *          On entry, LDA specifies the first dimension of A as declared
 *          in the calling (sub) program. LDA must be at least
 *          max( 1, N ).
 */
void csyr(
    const char* uplo,
    const INT n,
    const c64 alpha,
    const c64* restrict X,
    const INT incx,
    c64* restrict A,
    const INT lda)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    INT i, info, ix, j, jx, kx;
    c64 temp;

    info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        info = 1;
    } else if (n < 0) {
        info = 2;
    } else if (incx == 0) {
        info = 5;
    } else if (lda < (1 > n ? 1 : n)) {
        info = 7;
    }
    if (info != 0) {
        xerbla("CSYR  ", -info);
        return;
    }

    if ((n == 0) || (alpha == ZERO)) {
        return;
    }

    if (incx <= 0) {
        kx = -(n - 1) * incx;
    } else if (incx != 1) {
        kx = 0;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        if (incx == 1) {
            for (j = 0; j < n; j++) {
                if (X[j] != ZERO) {
                    temp = alpha * X[j];
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = A[i + j * lda] + X[i] * temp;
                    }
                }
            }
        } else {
            jx = kx;
            for (j = 0; j < n; j++) {
                if (X[jx] != ZERO) {
                    temp = alpha * X[jx];
                    ix = kx;
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = A[i + j * lda] + X[ix] * temp;
                        ix = ix + incx;
                    }
                }
                jx = jx + incx;
            }
        }
    } else {

        if (incx == 1) {
            for (j = 0; j < n; j++) {
                if (X[j] != ZERO) {
                    temp = alpha * X[j];
                    for (i = j; i < n; i++) {
                        A[i + j * lda] = A[i + j * lda] + X[i] * temp;
                    }
                }
            }
        } else {
            jx = kx;
            for (j = 0; j < n; j++) {
                if (X[jx] != ZERO) {
                    temp = alpha * X[jx];
                    ix = jx;
                    for (i = j; i < n; i++) {
                        A[i + j * lda] = A[i + j * lda] + X[ix] * temp;
                        ix = ix + incx;
                    }
                }
                jx = jx + incx;
            }
        }
    }
}
