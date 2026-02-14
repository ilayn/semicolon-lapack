/**
 * @file dlarf1l.c
 * @brief DLARF1L applies an elementary reflector to a general rectangular
 *        matrix assuming v(lastv) = 1 where lastv is the last non-zero element.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLARF1L applies a real elementary reflector H to a real m by n matrix
 * C, from either the left or the right. H is represented in the form
 *
 *       H = I - tau * v * v**T
 *
 * where tau is a real scalar and v is a real vector, with v(lastv) = 1
 * (where lastv is the last non-zero element).
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * @param[in]     side   'L': form H * C; 'R': form C * H
 * @param[in]     m      The number of rows of C.
 * @param[in]     n      The number of columns of C.
 * @param[in]     v      The vector v in the representation of H.
 *                       Dimension (1+(m-1)*|incv|) if side='L',
 *                       or (1+(n-1)*|incv|) if side='R'.
 * @param[in]     incv   The increment between elements of v. incv != 0.
 * @param[in]     tau    The value tau in the representation of H.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On exit, overwritten by H*C or C*H.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if side='L', (m) if side='R'.
 */
void dlarf1l(const char* side, const int m, const int n,
             const f64* restrict v, const int incv,
             const f64 tau,
             f64* restrict C, const int ldc,
             f64* restrict work)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    int applyleft;
    int firstv, lastv, lastc, i;

    applyleft = (side[0] == 'L' || side[0] == 'l');
    firstv = 0;  /* 0-based */
    lastc = 0;

    if (tau != 0.0) {
        /* Set up variables for scanning V. LASTV begins at the end of V. */
        if (applyleft) {
            lastv = m;
        } else {
            lastv = n;
        }

        i = 0;  /* Start scanning from the beginning */

        /* Look for the first non-zero row in V (scanning from front). */
        while (lastv > firstv + 1 && v[i] == 0.0) {
            firstv++;
            i += incv;
        }

        if (applyleft) {
            /* Scan for the last non-zero column in C(0:lastv-1, :) */
            lastc = iladlc(lastv, n, C, ldc);
        } else {
            /* Scan for the last non-zero row in C(:, 0:lastv-1) */
            lastc = iladlr(m, lastv, C, ldc);
        }
    }

    if (lastc == 0) {
        return;
    }

    if (applyleft) {
        /* Form H * C */
        if (lastv > 0) {
            if (lastv == firstv + 1) {
                /* v has only one element = 1, so H*C = (1 - tau)*C(firstv, :) */
                cblas_dscal(lastc, ONE - tau, &C[firstv], ldc);
            } else {
                /* w(0:lastc-1) := C(firstv:lastv-2, 0:lastc-1)^T * v(i:) */
                cblas_dgemv(CblasColMajor, CblasTrans,
                            lastv - firstv - 1, lastc, ONE,
                            &C[firstv], ldc,
                            &v[i], incv,
                            ZERO, work, 1);

                /* w += C(lastv-1, 0:lastc-1)^T (since v(lastv-1) = 1) */
                cblas_daxpy(lastc, ONE, &C[lastv - 1], ldc, work, 1);

                /* C(lastv-1, 0:lastc-1) -= tau * w^T */
                cblas_daxpy(lastc, -tau, work, 1, &C[lastv - 1], ldc);

                /* C(firstv:lastv-2, 0:lastc-1) -= tau * v(i:) * w^T */
                cblas_dger(CblasColMajor, lastv - firstv - 1, lastc,
                           -tau, &v[i], incv, work, 1,
                           &C[firstv], ldc);
            }
        }
    } else {
        /* Form C * H */
        if (lastv > 0) {
            if (lastv == firstv + 1) {
                /* v has only one element = 1, so C*H = (1 - tau)*C(:, 0) */
                cblas_dscal(lastc, ONE - tau, C, 1);
            } else {
                /* w(0:lastc-1) := C(0:lastc-1, firstv:lastv-2) * v(i:) */
                cblas_dgemv(CblasColMajor, CblasNoTrans,
                            lastc, lastv - firstv - 1, ONE,
                            &C[firstv * ldc], ldc,
                            &v[i], incv,
                            ZERO, work, 1);

                /* w += C(0:lastc-1, lastv-1) (since v(lastv-1) = 1) */
                cblas_daxpy(lastc, ONE, &C[(lastv - 1) * ldc], 1, work, 1);

                /* C(0:lastc-1, lastv-1) -= tau * w */
                cblas_daxpy(lastc, -tau, work, 1, &C[(lastv - 1) * ldc], 1);

                /* C(0:lastc-1, firstv:lastv-2) -= tau * w * v(i:)^T */
                cblas_dger(CblasColMajor, lastc, lastv - firstv - 1,
                           -tau, work, 1, &v[i], incv,
                           &C[firstv * ldc], ldc);
            }
        }
    }
}
