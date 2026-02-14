/**
 * @file dlarf1f.c
 * @brief DLARF1F applies an elementary reflector to a general rectangular
 *        matrix assuming v(1) = 1.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLARF1F applies a real elementary reflector H to a real m by n matrix
 * C, from either the left or the right. H is represented in the form
 *
 *       H = I - tau * v * v**T
 *
 * where tau is a real scalar and v is a real vector, with v(1) = 1
 * (not stored).
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * @param[in]     side   'L': form H * C; 'R': form C * H
 * @param[in]     m      The number of rows of C.
 * @param[in]     n      The number of columns of C.
 * @param[in]     v      The vector v in the representation of H.
 *                       v(0) is not referenced (implicitly 1).
 *                       Dimension (1+(m-1)*|incv|) if side='L',
 *                       or (1+(n-1)*|incv|) if side='R'.
 * @param[in]     incv   The increment between elements of v. incv != 0.
 * @param[in]     tau    The value tau in the representation of H.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On exit, overwritten by H*C or C*H.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if side='L', (m) if side='R'.
 */
void dlarf1f(const char* side, const int m, const int n,
             const f64 * const restrict v, const int incv,
             const f64 tau,
             f64 * const restrict C, const int ldc,
             f64 * const restrict work)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    int applyleft;
    int lastv, lastc, i;

    applyleft = (side[0] == 'L' || side[0] == 'l');
    lastv = 1;
    lastc = 0;

    if (tau != 0.0) {
        /* Set up variables for scanning V. LASTV begins pointing to the end of V. */
        if (applyleft) {
            lastv = m;
        } else {
            lastv = n;
        }

        if (incv > 0) {
            i = (lastv - 1) * incv;  /* 0-based: last element index */
        } else {
            i = 0;
        }

        /* Look for the last non-zero row in V.
         * Since v(0) = 1 implicitly, don't access it. */
        while (lastv > 1 && v[i] == 0.0) {
            lastv--;
            i -= incv;
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
        if (lastv == 1) {
            /* v = [1], so H*C = (1 - tau)*C(0, 0:lastc-1) */
            cblas_dscal(lastc, ONE - tau, C, ldc);
        } else {
            /* w(0:lastc-1) := C(1:lastv-1, 0:lastc-1)^T * v(incv:) */
            cblas_dgemv(CblasColMajor, CblasTrans,
                        lastv - 1, lastc, ONE,
                        &C[1], ldc,
                        &v[incv], incv,
                        ZERO, work, 1);

            /* w(0:lastc-1) += C(0, 0:lastc-1)^T (since v(0) = 1) */
            cblas_daxpy(lastc, ONE, C, ldc, work, 1);

            /* C(0, 0:lastc-1) -= tau * w */
            cblas_daxpy(lastc, -tau, work, 1, C, ldc);

            /* C(1:lastv-1, 0:lastc-1) -= tau * v(incv:) * w^T */
            cblas_dger(CblasColMajor, lastv - 1, lastc,
                       -tau, &v[incv], incv, work, 1,
                       &C[1], ldc);
        }
    } else {
        /* Form C * H */
        if (lastv == 1) {
            /* v = [1], so C*H = (1 - tau)*C(0:lastc-1, 0) */
            cblas_dscal(lastc, ONE - tau, C, 1);
        } else {
            /* w(0:lastc-1) := C(0:lastc-1, 1:lastv-1) * v(incv:) */
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        lastc, lastv - 1, ONE,
                        &C[ldc], ldc,
                        &v[incv], incv,
                        ZERO, work, 1);

            /* w(0:lastc-1) += C(0:lastc-1, 0) (since v(0) = 1) */
            cblas_daxpy(lastc, ONE, C, 1, work, 1);

            /* C(0:lastc-1, 0) -= tau * w */
            cblas_daxpy(lastc, -tau, work, 1, C, 1);

            /* C(0:lastc-1, 1:lastv-1) -= tau * w * v(incv:)^T */
            cblas_dger(CblasColMajor, lastc, lastv - 1,
                       -tau, work, 1, &v[incv], incv,
                       &C[ldc], ldc);
        }
    }
}
