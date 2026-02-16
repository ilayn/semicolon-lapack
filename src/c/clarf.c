/**
 * @file clarf.c
 * @brief CLARF applies an elementary reflector to a general rectangular matrix.
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
 * CLARF applies a complex elementary reflector H to a complex m by n matrix
 * C, from either the left or the right. H is represented in the form
 *
 *       H = I - tau * v * v**H
 *
 * where tau is a complex scalar and v is a complex vector.
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
 * @param[in,out] C      Single complex array, dimension (ldc, n).
 *                       On exit, overwritten by H*C or C*H.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Workspace, dimension (n) if side='L', (m) if side='R'.
 */
void clarf(const char* side, const int m, const int n,
           const c64* restrict v, const int incv,
           const c64 tau,
           c64* restrict C, const int ldc,
           c64* restrict work)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    int applyleft;
    int lastv, lastc, i;

    applyleft = (side[0] == 'L' || side[0] == 'l');
    lastv = 0;
    lastc = 0;

    if (tau != ZERO) {
        /* Set up variables for scanning V. LASTV begins pointing to the end of V. */
        if (applyleft) {
            lastv = m;
        } else {
            lastv = n;
        }

        if (incv > 0) {
            i = (lastv - 1) * incv;  /* 0-based index of last element */
        } else {
            i = 0;
        }

        /* Look for the last non-zero row in V. */
        while (lastv > 0 && v[i] == ZERO) {
            lastv--;
            i -= incv;
        }

        if (applyleft) {
            /* Scan for the last non-zero column in C(0:lastv-1, :) */
            lastc = ilaclc(lastv, n, C, ldc);
        } else {
            /* Scan for the last non-zero row in C(:, 0:lastv-1) */
            lastc = ilaclr(m, lastv, C, ldc);
        }
    }

    /* Note that lastc == 0 renders the BLAS operations null */
    if (applyleft) {
        /* Form H * C */
        if (lastv > 0) {
            /* w(0:lastc-1) := C(0:lastv-1, 0:lastc-1)**H * v(0:lastv-1) */
            cblas_cgemv(CblasColMajor, CblasConjTrans,
                        lastv, lastc, &ONE, C, ldc,
                        v, incv, &ZERO, work, 1);

            /* C(0:lastv-1, 0:lastc-1) -= tau * v(0:lastv-1) * w**H */
            const c64 neg_tau = -tau;
            cblas_cgerc(CblasColMajor, lastv, lastc,
                        &neg_tau, v, incv, work, 1, C, ldc);
        }
    } else {
        /* Form C * H */
        if (lastv > 0) {
            /* w(0:lastc-1) := C(0:lastc-1, 0:lastv-1) * v(0:lastv-1) */
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        lastc, lastv, &ONE, C, ldc,
                        v, incv, &ZERO, work, 1);

            /* C(0:lastc-1, 0:lastv-1) -= tau * w * v**H */
            const c64 neg_tau = -tau;
            cblas_cgerc(CblasColMajor, lastc, lastv,
                        &neg_tau, work, 1, v, incv, C, ldc);
        }
    }
}
