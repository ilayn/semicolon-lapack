/**
 * @file clarf1l.c
 * @brief CLARF1L applies an elementary reflector to a general rectangular
 *        matrix assuming v(lastv) = 1 where lastv is the last non-zero element.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/** @cond */
static int ilaclc(
    const int m,
    const int n,
    const c64* restrict A,
    const int lda)
{
    const c64 zero = CMPLXF(0.0f, 0.0f);
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

static int ilaclr(
    const int m,
    const int n,
    const c64* restrict A,
    const int lda)
{
    const c64 zero = CMPLXF(0.0f, 0.0f);
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
/** @endcond */

/**
 * CLARF1L applies a complex elementary reflector H to a complex m by n matrix
 * C, from either the left or the right. H is represented in the form
 *
 *       H = I - tau * v * v**H
 *
 * where tau is a complex scalar and v is a complex vector, with v(lastv) = 1
 * (where lastv is the last non-zero element).
 *
 * If tau = 0, then H is taken to be the unit matrix.
 *
 * To apply H**H (the conjugate transpose of H), supply conjg(tau) instead
 * tau.
 *
 * @param[in]     side   'L': form H * C; 'R': form C * H
 * @param[in]     m      The number of rows of C.
 * @param[in]     n      The number of columns of C.
 * @param[in]     v      The vector v in the representation of H.
 *                       Dimension (1+(m-1)*|incv|) if side='L',
 *                       or (1+(n-1)*|incv|) if side='R'.
 * @param[in]     incv   The increment between elements of v. incv > 0.
 * @param[in]     tau    The value tau in the representation of H.
 * @param[in,out] C      Complex array, dimension (ldc, n).
 *                       On exit, overwritten by H*C or C*H.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if side='L', (m) if side='R'.
 */
void clarf1l(const char* side, const int m, const int n,
             const c64* restrict v, const int incv,
             const c64 tau,
             c64* restrict C, const int ldc,
             c64* restrict work)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    c64 neg_tau;
    c64 one_minus_tau;
    int applyleft;
    int firstv, lastv, lastc, i, j;

    applyleft = (side[0] == 'L' || side[0] == 'l');
    firstv = 0;  /* 0-based */
    lastc = 0;

    if (tau != ZERO) {
        /* Set up variables for scanning V. LASTV begins at the end of V. */
        if (applyleft) {
            lastv = m;
        } else {
            lastv = n;
        }

        i = 0;

        /* Look for the first non-zero row in V (scanning from front). */
        while (lastv > firstv + 1 && v[i] == ZERO) {
            firstv++;
            i += incv;
        }

        if (applyleft) {
            /* Scan for the last non-zero column in C(0:lastv-1, :) */
            lastc = ilaclc(lastv, n, C, ldc);
        } else {
            /* Scan for the last non-zero row in C(:, 0:lastv-1) */
            lastc = ilaclr(m, lastv, C, ldc);
        }
    }

    if (lastc == 0) {
        return;
    }

    if (applyleft) {
        /*
         * Form  H * C
         */
        if (lastv == firstv + 1) {
            /*
             * C(lastv-1,0:lastc-1) := ( 1 - tau ) * C(lastv-1,0:lastc-1)
             */
            one_minus_tau = ONE - tau;
            cblas_cscal(lastc, &one_minus_tau, &C[lastv - 1], ldc);
        } else {
            /*
             * w(0:lastc-1) := C(firstv:lastv-2,0:lastc-1)**H * v(firstv:lastv-2)
             */
            cblas_cgemv(CblasColMajor, CblasConjTrans,
                        lastv - firstv - 1, lastc, &ONE,
                        &C[firstv], ldc,
                        &v[i], incv, &ZERO,
                        work, 1);

            /*
             * w(0:lastc-1) += conj(C(lastv-1,0:lastc-1))
             */
            for (j = 0; j < lastc; j++) {
                work[j] = work[j] + conjf(C[(lastv - 1) + j * ldc]);
            }

            /*
             * C(lastv-1,0:lastc-1) += - tau * conj(w(0:lastc-1))
             */
            for (j = 0; j < lastc; j++) {
                C[(lastv - 1) + j * ldc] = C[(lastv - 1) + j * ldc]
                                            - tau * conjf(work[j]);
            }

            /*
             * C(firstv:lastv-2,0:lastc-1) += - tau * v(firstv:lastv-2) * w(0:lastc-1)**H
             */
            neg_tau = -tau;
            cblas_cgerc(CblasColMajor, lastv - firstv - 1, lastc,
                        &neg_tau, &v[i], incv,
                        work, 1, &C[firstv], ldc);
        }
    } else {
        /*
         * Form  C * H
         */
        if (lastv == firstv + 1) {
            /*
             * C(0:lastc-1,lastv-1) := ( 1 - tau ) * C(0:lastc-1,lastv-1)
             */
            one_minus_tau = ONE - tau;
            cblas_cscal(lastc, &one_minus_tau, &C[(lastv - 1) * ldc], 1);
        } else {
            /*
             * w(0:lastc-1) := C(0:lastc-1,firstv:lastv-2) * v(firstv:lastv-2)
             */
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        lastc, lastv - firstv - 1, &ONE,
                        &C[firstv * ldc], ldc,
                        &v[i], incv, &ZERO,
                        work, 1);

            /*
             * w(0:lastc-1) += C(0:lastc-1,lastv-1)
             */
            cblas_caxpy(lastc, &ONE, &C[(lastv - 1) * ldc], 1, work, 1);

            /*
             * C(0:lastc-1,lastv-1) += - tau * w(0:lastc-1)
             */
            neg_tau = -tau;
            cblas_caxpy(lastc, &neg_tau, work, 1, &C[(lastv - 1) * ldc], 1);

            /*
             * C(0:lastc-1,firstv:lastv-2) += - tau * w(0:lastc-1) * v(firstv:lastv-2)**H
             */
            cblas_cgerc(CblasColMajor, lastc, lastv - firstv - 1,
                        &neg_tau, work, 1, &v[i], incv,
                        &C[firstv * ldc], ldc);
        }
    }
}
