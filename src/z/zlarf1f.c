/**
 * @file zlarf1f.c
 * @brief ZLARF1F applies an elementary reflector to a general rectangular
 *        matrix assuming v(1) = 1.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARF1F applies a complex elementary reflector H to a complex m by n matrix
 * C, from either the left or the right. H is represented in the form
 *
 *       H = I - tau * v * v**H
 *
 * where tau is a complex scalar and v is a complex vector, with v(1) = 1
 * (not stored).
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * To apply H**H, supply conjg(tau) instead tau.
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
 * @param[in,out] C      Complex array, dimension (ldc, n).
 *                       On exit, overwritten by H*C or C*H.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if side='L', (m) if side='R'.
 */
void zlarf1f(const char* side, const INT m, const INT n,
             const c128* restrict v, const INT incv,
             const c128 tau,
             c128* restrict C, const INT ldc,
             c128* restrict work)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);
    c128 neg_tau;
    c128 one_minus_tau;
    INT applyleft;
    INT lastv, lastc, i, j;

    /* Quick return if C is empty */
    if (m <= 0 || n <= 0) {
        return;
    }

    applyleft = (side[0] == 'L' || side[0] == 'l');
    lastv = 1;
    lastc = 0;

    if (tau != ZERO) {
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
        while (lastv > 1 && v[i] == ZERO) {
            lastv--;
            i -= incv;
        }

        if (applyleft) {
            /* Scan for the last non-zero column in C(0:lastv-1, :) */
            lastc = ilazlc(lastv, n, C, ldc);
        } else {
            /* Scan for the last non-zero row in C(:, 0:lastv-1) */
            lastc = ilazlr(m, lastv, C, ldc);
        }

        /* Set index for V. If INCV < 0, then I points to the end of V.
         * For INCV > 0, set I to point to V(2) */
        if (incv > 0) {
            i = incv;
        }
    }

    if (lastc == 0) {
        return;
    }

    if (applyleft) {
        /*
         * Form  H * C
         */
        if (lastv == 1) {
            /* v = [1], so H*C = (1 - tau)*C(0, 0:lastc-1) */
            one_minus_tau = ONE - tau;
            cblas_zscal(lastc, &one_minus_tau, C, ldc);
        } else {
            /*
             * w(0:lastc-1) := C(1:lastv-1, 0:lastc-1)**H * v(i:)
             */
            cblas_zgemv(CblasColMajor, CblasConjTrans,
                        lastv - 1, lastc, &ONE,
                        &C[1], ldc,
                        &v[i], incv,
                        &ZERO, work, 1);

            /*
             * w(0:lastc-1) += conj(C(0, 0:lastc-1))
             */
            for (j = 0; j < lastc; j++) {
                work[j] = work[j] + conj(C[j * ldc]);
            }

            /*
             * C(0, 0:lastc-1) -= tau * conj(w)
             */
            for (j = 0; j < lastc; j++) {
                C[j * ldc] = C[j * ldc] - tau * conj(work[j]);
            }

            /*
             * C(1:lastv-1, 0:lastc-1) -= tau * v(i:) * w(0:lastc-1)**H
             */
            neg_tau = -tau;
            cblas_zgerc(CblasColMajor, lastv - 1, lastc,
                        &neg_tau, &v[i], incv, work, 1,
                        &C[1], ldc);
        }
    } else {
        /*
         * Form  C * H
         */
        if (lastv == 1) {
            /* v = [1], so C*H = (1 - tau)*C(0:lastc-1, 0) */
            one_minus_tau = ONE - tau;
            cblas_zscal(lastc, &one_minus_tau, C, 1);
        } else {
            /*
             * w(0:lastc-1) := C(0:lastc-1, 1:lastv-1) * v(i:)
             */
            cblas_zgemv(CblasColMajor, CblasNoTrans,
                        lastc, lastv - 1, &ONE,
                        &C[ldc], ldc,
                        &v[i], incv,
                        &ZERO, work, 1);

            /*
             * w(0:lastc-1) += C(0:lastc-1, 0)
             */
            cblas_zaxpy(lastc, &ONE, C, 1, work, 1);

            /*
             * C(0:lastc-1, 0) -= tau * w
             */
            neg_tau = -tau;
            cblas_zaxpy(lastc, &neg_tau, work, 1, C, 1);

            /*
             * C(0:lastc-1, 1:lastv-1) -= tau * w * v(i:)**H
             */
            cblas_zgerc(CblasColMajor, lastc, lastv - 1,
                        &neg_tau, work, 1, &v[i], incv,
                        &C[ldc], ldc);
        }
    }
}
