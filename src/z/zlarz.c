/**
 * @file zlarz.c
 * @brief ZLARZ applies an elementary reflector (as returned by ZTZRZF) to a
 *        general matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARZ applies a complex elementary reflector H to a complex M-by-N
 * matrix C, from either the left or the right. H is represented in the
 * form
 *
 *       H = I - tau * v * v**H
 *
 * where tau is a complex scalar and v is a complex vector.
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * To apply H**H (the conjugate transpose of H), supply conjg(tau) instead
 * tau.
 *
 * H is a product of k elementary reflectors as returned by ZTZRZF.
 *
 * @param[in]     side  CHARACTER*1
 *                      = 'L': form  H * C
 *                      = 'R': form  C * H
 * @param[in]     m     The number of rows of the matrix C.
 * @param[in]     n     The number of columns of the matrix C.
 * @param[in]     l     The number of entries of the vector V containing
 *                      the meaningful part of the Householder vectors.
 *                      If side = "L", m >= l >= 0, if side = "R", n >= l >= 0.
 * @param[in]     v     Complex*16 array, dimension (1+(l-1)*abs(incv)).
 *                      The vector v in the representation of H as returned by
 *                      ZTZRZF. V is not used if tau = 0.
 * @param[in]     incv  The increment between elements of v. incv != 0.
 * @param[in]     tau   The value tau in the representation of H.
 * @param[in,out] C     Complex*16 array, dimension (ldc, n).
 *                      On entry, the M-by-N matrix C.
 *                      On exit, C is overwritten by the matrix H * C if side = "L",
 *                      or C * H if side = 'R'.
 * @param[in]     ldc   The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work  Complex*16 array, dimension
 *                      (n) if side = 'L'
 *                      or (m) if side = 'R'.
 */
void zlarz(const char* side, const int m, const int n, const int l,
           const c128* restrict v, const int incv,
           const c128 tau,
           c128* restrict C, const int ldc,
           c128* restrict work)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

    if (side[0] == 'L' || side[0] == 'l') {

        /* Form  H * C */

        if (tau != ZERO) {

            /* w(1:n) = conjg( C(1, 1:n) ) */
            cblas_zcopy(n, C, ldc, work, 1);
            zlacgv(n, work, 1);

            /* w(1:n) = conjg( w(1:n) + C(m-l+1:m, 1:n)**H * v(1:l) ) */
            cblas_zgemv(CblasColMajor, CblasConjTrans, l, n, &ONE,
                        &C[(m - l) + 0 * ldc], ldc,
                        v, incv, &ONE, work, 1);
            zlacgv(n, work, 1);

            /* C(1, 1:n) = C(1, 1:n) - tau * w(1:n) */
            {
                const c128 neg_tau = -tau;
                cblas_zaxpy(n, &neg_tau, work, 1, C, ldc);
            }

            /* C(m-l+1:m, 1:n) = C(m-l+1:m, 1:n) - tau * v(1:l) * w(1:n)**T */
            {
                const c128 neg_tau = -tau;
                cblas_zgeru(CblasColMajor, l, n, &neg_tau, v, incv,
                            work, 1, &C[(m - l) + 0 * ldc], ldc);
            }
        }

    } else {

        /* Form  C * H */

        if (tau != ZERO) {

            /* w(1:m) = C(1:m, 1) */
            cblas_zcopy(m, C, 1, work, 1);

            /* w(1:m) = w(1:m) + C(1:m, n-l+1:n) * v(1:l) */
            cblas_zgemv(CblasColMajor, CblasNoTrans, m, l, &ONE,
                        &C[0 + (n - l) * ldc], ldc,
                        v, incv, &ONE, work, 1);

            /* C(1:m, 1) = C(1:m, 1) - tau * w(1:m) */
            {
                const c128 neg_tau = -tau;
                cblas_zaxpy(m, &neg_tau, work, 1, C, 1);
            }

            /* C(1:m, n-l+1:n) = C(1:m, n-l+1:n) - tau * w(1:m) * v(1:l)**H */
            {
                const c128 neg_tau = -tau;
                cblas_zgerc(CblasColMajor, m, l, &neg_tau, work, 1,
                            v, incv, &C[0 + (n - l) * ldc], ldc);
            }
        }

    }
}
