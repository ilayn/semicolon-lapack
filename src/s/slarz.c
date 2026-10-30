/**
 * @file slarz.c
 * @brief SLARZ applies an elementary reflector (as returned by STZRZF) to a
 *        general matrix.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLARZ applies a real elementary reflector H to a real M-by-N
 * matrix C, from either the left or the right. H is represented in the
 * form
 *
 *       H = I - tau * v * v**T
 *
 * where tau is a real scalar and v is a real vector.
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * H is a product of k elementary reflectors as returned by STZRZF.
 *
 * @param[in]     side  CHARACTER*1
 *                      = 'L': form  H * C
 *                      = 'R': form  C * H
 * @param[in]     m     The number of rows of the matrix C.
 * @param[in]     n     The number of columns of the matrix C.
 * @param[in]     l     The number of entries of the vector V containing
 *                      the meaningful part of the Householder vectors.
 *                      If side = "L", m >= l >= 0, if side = "R", n >= l >= 0.
 * @param[in]     v     Double precision array, dimension (1+(l-1)*abs(incv)).
 *                      The vector v in the representation of H as returned by
 *                      STZRZF. V is not used if tau = 0.
 * @param[in]     incv  The increment between elements of v. incv != 0.
 * @param[in]     tau   The value tau in the representation of H.
 * @param[in,out] C     Double precision array, dimension (ldc, n).
 *                      On entry, the M-by-N matrix C.
 *                      On exit, C is overwritten by the matrix H * C if side = "L",
 *                      or C * H if side = 'R'.
 * @param[in]     ldc   The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work  Double precision array, dimension
 *                      (n) if side = 'L'
 *                      or (m) if side = 'R'.
 */
void slarz(const char* side, const INT m, const INT n, const INT l,
           const f32* restrict v, const INT incv,
           const f32 tau,
           f32* restrict C, const INT ldc,
           f32* restrict work)
{
    if (side[0] == 'L' || side[0] == 'l') {

        /* Form  H * C */

        if (tau != 0.0f) {

            /* w(1:n) = C(1, 1:n) */
            cblas_scopy(n, C, ldc, work, 1);

            /* w(1:n) = w(1:n) + C(m-l+1:m, 1:n)**T * v(1:l) */
            cblas_sgemv(CblasColMajor, CblasTrans, l, n, 1.0f,
                        &C[(m - l) + 0 * ldc], ldc,
                        v, incv, 1.0f, work, 1);

            /* C(1, 1:n) = C(1, 1:n) - tau * w(1:n) */
            cblas_saxpy(n, -tau, work, 1, C, ldc);

            /* C(m-l+1:m, 1:n) = C(m-l+1:m, 1:n) - tau * v(1:l) * w(1:n)**T */
            cblas_sger(CblasColMajor, l, n, -tau, v, incv,
                       work, 1, &C[(m - l) + 0 * ldc], ldc);
        }

    } else {

        /* Form  C * H */

        if (tau != 0.0f) {

            /* w(1:m) = C(1:m, 1) */
            cblas_scopy(m, C, 1, work, 1);

            /* w(1:m) = w(1:m) + C(1:m, n-l+1:n) * v(1:l) */
            cblas_sgemv(CblasColMajor, CblasNoTrans, m, l, 1.0f,
                        &C[0 + (n - l) * ldc], ldc,
                        v, incv, 1.0f, work, 1);

            /* C(1:m, 1) = C(1:m, 1) - tau * w(1:m) */
            cblas_saxpy(m, -tau, work, 1, C, 1);

            /* C(1:m, n-l+1:n) = C(1:m, n-l+1:n) - tau * w(1:m) * v(1:l)**T */
            cblas_sger(CblasColMajor, m, l, -tau, work, 1,
                       v, incv, &C[0 + (n - l) * ldc], ldc);
        }

    }
}
