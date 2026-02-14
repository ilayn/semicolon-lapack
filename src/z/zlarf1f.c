/**
 * @file zlarf1f.c
 * @brief ZLARF1F applies an elementary reflector to a general rectangular
 *        matrix assuming v(1) = 1.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

static int ilazlc(
    const int m,
    const int n,
    const c128* restrict A,
    const int lda)
{
    const c128 zero = CMPLX(0.0, 0.0);
    int i, j;

    if (n == 0) {
        return 0;
    } else if (A[0 + (n - 1) * lda] != zero || A[(m - 1) + (n - 1) * lda] != zero) {
        return n;
    } else {
        for (j = n - 1; j >= 0; j--) {
            for (i = 0; i < m; i++) {
                if (A[i + j * lda] != zero) {
                    return j + 1;
                }
            }
        }
        return 0;
    }
}

static int ilazlr(
    const int m,
    const int n,
    const c128* restrict A,
    const int lda)
{
    const c128 zero = CMPLX(0.0, 0.0);
    int i, j, result;

    if (m == 0) {
        return 0;
    } else if (A[(m - 1) + 0 * lda] != zero || A[(m - 1) + (n - 1) * lda] != zero) {
        return m;
    } else {
        result = 0;
        for (j = 0; j < n; j++) {
            i = m - 1;
            while (i >= 0 && A[i + j * lda] == zero) {
                i--;
            }
            if (i + 1 > result) {
                result = i + 1;
            }
        }
        return result;
    }
}

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
void zlarf1f(const char* side, const int m, const int n,
             const c128* restrict v, const int incv,
             const c128 tau,
             c128* restrict C, const int ldc,
             c128* restrict work)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);
    c128 neg_tau;
    c128 one_minus_tau;
    int applyleft;
    int lastv, lastc, i;

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
             * w(0:lastc-1) := C(1:lastv-1, 0:lastc-1)**H * v(incv:)
             */
            cblas_zgemv(CblasColMajor, CblasConjTrans,
                        lastv - 1, lastc, &ONE,
                        &C[1], ldc,
                        &v[incv], incv,
                        &ZERO, work, 1);

            /*
             * w(0:lastc-1) += conj(C(0, 0:lastc-1))
             */
            for (i = 0; i < lastc; i++) {
                work[i] = work[i] + conj(C[i * ldc]);
            }

            /*
             * C(0, 0:lastc-1) -= tau * conj(w)
             */
            for (i = 0; i < lastc; i++) {
                C[i * ldc] = C[i * ldc] - tau * conj(work[i]);
            }

            /*
             * C(1:lastv-1, 0:lastc-1) -= tau * v(incv:) * w(0:lastc-1)**H
             */
            neg_tau = -tau;
            cblas_zgerc(CblasColMajor, lastv - 1, lastc,
                        &neg_tau, &v[incv], incv, work, 1,
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
             * w(0:lastc-1) := C(0:lastc-1, 1:lastv-1) * v(incv:)
             */
            cblas_zgemv(CblasColMajor, CblasNoTrans,
                        lastc, lastv - 1, &ONE,
                        &C[ldc], ldc,
                        &v[incv], incv,
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
             * C(0:lastc-1, 1:lastv-1) -= tau * w * v(incv:)**H
             */
            cblas_zgerc(CblasColMajor, lastc, lastv - 1,
                        &neg_tau, work, 1, &v[incv], incv,
                        &C[ldc], ldc);
        }
    }
}
