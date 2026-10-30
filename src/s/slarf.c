/**
 * @file slarf.c
 * @brief SLARF applies an elementary reflector to a general rectangular matrix.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLARF applies a real elementary reflector H to a real m by n matrix
 * C, from either the left or the right. H is represented in the form
 *
 *       H = I - tau * v * v**T
 *
 * where tau is a real scalar and v is a real vector.
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
void slarf(const char* side, const INT m, const INT n,
           const f32* restrict v, const INT incv,
           const f32 tau,
           f32* restrict C, const INT ldc,
           f32* restrict work)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    INT applyleft;
    INT lastv, lastc, i;

    applyleft = (side[0] == 'L' || side[0] == 'l');
    lastv = 0;
    lastc = 0;

    if (tau != 0.0f) {
        /* Set up variables for scanning V. LASTV begins pointing to the end of V. */
        if (applyleft) {
            lastv = m;
        } else {
            lastv = n;
        }

        if (incv > 0) {
            i = (lastv - 1) * incv;
        } else {
            i = 0;
        }

        /* Look for the last non-zero row in V. */
        while (lastv > 0 && v[i] == 0.0f) {
            lastv--;
            i -= incv;
        }

        if (applyleft) {
            /* Scan for the last non-zero column in C(0:lastv-1, :) */
            lastc = ilaslc(lastv, n, C, ldc);
        } else {
            /* Scan for the last non-zero row in C(:, 0:lastv-1) */
            lastc = ilaslr(m, lastv, C, ldc);
        }

        /* Set index for V. If INCV < 0, then I points to the end of V.
         * For INCV > 0, set I = 0 */
        if (incv > 0) {
            i = 0;
        }
    }

    /* Note that lastc == 0 renders the BLAS operations null; no special
     * case is needed at this level. */
    if (applyleft) {
        /* Form H * C */
        if (lastv > 0) {
            /* w(0:lastc-1,0) := C(0:lastv-1,0:lastc-1)**T * v(0:lastv-1,0) */
            cblas_sgemv(CblasColMajor, CblasTrans,
                        lastv, lastc, ONE, C, ldc,
                        &v[i], incv, ZERO, work, 1);

            /* C(0:lastv-1,0:lastc-1) := C(...) - v(0:lastv-1,0) * w(0:lastc-1,0)**T */
            cblas_sger(CblasColMajor, lastv, lastc,
                       -tau, &v[i], incv, work, 1, C, ldc);
        }
    } else {
        /* Form C * H */
        if (lastv > 0) {
            /* w(0:lastc-1,0) := C(0:lastc-1,0:lastv-1) * v(0:lastv-1,0) */
            cblas_sgemv(CblasColMajor, CblasNoTrans,
                        lastc, lastv, ONE, C, ldc,
                        &v[i], incv, ZERO, work, 1);

            /* C(0:lastc-1,0:lastv-1) := C(...) - w(0:lastc-1,0) * v(0:lastv-1,0)**T */
            cblas_sger(CblasColMajor, lastc, lastv,
                       -tau, work, 1, &v[i], incv, C, ldc);
        }
    }
}
