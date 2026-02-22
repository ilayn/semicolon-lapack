/**
 * @file clarfy.c
 * @brief CLARFY applies an elementary reflector to a Hermitian matrix.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CLARFY applies an elementary reflector, or Householder matrix, H,
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
void clarfy(const char* uplo, const INT n,
            const c64* restrict V,
            const INT incv, const c64 tau,
            c64* restrict C,
            const INT ldc, c64* restrict work)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 HALF = CMPLXF(0.5f, 0.0f);
    c64 alpha;
    c64 neg_tau;
    c64 dotc;
    CBLAS_UPLO cblas_uplo;

    if (tau == ZERO) {
        return;
    }

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    /* Form w := C * v */
    cblas_chemv(CblasColMajor, cblas_uplo, n, &ONE, C, ldc, V, incv,
                &ZERO, work, 1);

    /* alpha := -0.5 * tau * dot(w, v) */
    cblas_cdotc_sub(n, work, 1, V, incv, &dotc);
    alpha = -HALF * tau * dotc;
    cblas_caxpy(n, &alpha, V, incv, work, 1);

    /* C := C - v * w' - w * v' */
    neg_tau = -tau;
    cblas_cher2(CblasColMajor, cblas_uplo, n, &neg_tau, V, incv, work, 1,
                C, ldc);
}
