/**
 * @file zlarfy.c
 * @brief ZLARFY applies an elementary reflector to a Hermitian matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARFY applies an elementary reflector, or Householder matrix, H,
 * to an N-by-N Hermitian matrix C, from both the left and the right.
 *
 * H is represented in the form
 *    H = I - tau * v * v'
 *
 * where tau is a scalar and v is a vector.
 *
 * If tau is zero, then H is taken to be the unit matrix.
 *
 * @param[in]     uplo   CHARACTER*1.
 *                       Specifies whether the upper or lower triangular part of the
 *                       Hermitian matrix C is stored.
 *                       = 'U': Upper triangle.
 *                       = 'L': Lower triangle.
 * @param[in]     n      The number of rows and columns of the matrix C. N >= 0.
 * @param[in]     V      Complex*16 array, dimension (1 + (N-1)*abs(INCV)).
 *                       The vector v as described above.
 * @param[in]     incv   The increment between successive elements of v. INCV must
 *                       not be zero.
 * @param[in]     tau    The value tau as described above.
 * @param[in,out] C      Complex*16 array, dimension (ldc, N).
 *                       On entry, the matrix C.
 *                       On exit, C is overwritten by H * C * H'.
 * @param[in]     ldc    The leading dimension of the array C. LDC >= max(1, N).
 * @param[out]    work   Complex*16 array, dimension (N).
 */
void zlarfy(const char* uplo, const int n,
            const double complex* const restrict V,
            const int incv, const double complex tau,
            double complex* const restrict C,
            const int ldc, double complex* const restrict work)
{
    const double complex ZERO = CMPLX(0.0, 0.0);
    const double complex ONE = CMPLX(1.0, 0.0);
    const double complex HALF = CMPLX(0.5, 0.0);
    double complex alpha;
    double complex neg_tau;
    double complex dotc;
    CBLAS_UPLO cblas_uplo;

    if (tau == ZERO) {
        return;
    }

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    /* Form w := C * v */
    cblas_zhemv(CblasColMajor, cblas_uplo, n, &ONE, C, ldc, V, incv,
                &ZERO, work, 1);

    /* alpha := -0.5 * tau * dot(w, v) */
    cblas_zdotc_sub(n, work, 1, V, incv, &dotc);
    alpha = -HALF * tau * dotc;
    cblas_zaxpy(n, &alpha, V, incv, work, 1);

    /* C := C - v * w' - w * v' */
    neg_tau = -tau;
    cblas_zher2(CblasColMajor, cblas_uplo, n, &neg_tau, V, incv, work, 1,
                C, ldc);
}
