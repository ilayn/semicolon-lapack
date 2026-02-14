/**
 * @file dlarfy.c
 * @brief DLARFY applies an elementary reflector to a symmetric matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLARFY applies an elementary reflector, or Householder matrix, H,
 * to an N-by-N symmetric matrix C, from both the left and the right.
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
 *                       symmetric matrix C is stored.
 *                       = 'U': Upper triangle.
 *                       = 'L': Lower triangle.
 * @param[in]     n      The number of rows and columns of the matrix C. N >= 0.
 * @param[in]     V      Double precision array, dimension (1 + (N-1)*abs(INCV)).
 *                       The vector v as described above.
 * @param[in]     incv   The increment between successive elements of v. INCV must
 *                       not be zero.
 * @param[in]     tau    The value tau as described above.
 * @param[in,out] C      Double precision array, dimension (ldc, N).
 *                       On entry, the matrix C.
 *                       On exit, C is overwritten by H * C * H'.
 * @param[in]     ldc    The leading dimension of the array C. LDC >= max(1, N).
 * @param[out]    work   Double precision array, dimension (N).
 */
void dlarfy(const char* uplo, const int n, const f64* restrict V,
            const int incv, const f64 tau, f64* restrict C,
            const int ldc, f64* restrict work)
{
    f64 alpha;
    CBLAS_UPLO cblas_uplo;

    if (tau == 0.0) {
        return;
    }

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    /* Form w := C * v */
    cblas_dsymv(CblasColMajor, cblas_uplo, n, 1.0, C, ldc, V, incv, 0.0, work, 1);

    /* alpha := -0.5 * tau * dot(w, v) */
    alpha = -0.5 * tau * cblas_ddot(n, work, 1, V, incv);
    cblas_daxpy(n, alpha, V, incv, work, 1);

    /* C := C - v * w' - w * v' */
    cblas_dsyr2(CblasColMajor, cblas_uplo, n, -tau, V, incv, work, 1, C, ldc);
}
