/**
 * @file cget01.c
 * @brief CGET01 reconstructs a matrix A from its L*U factorization and
 *        computes the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cget01(
    const INT m,
    const INT n,
    const c64* const restrict A,
    const INT lda,
    c64* const restrict AFAC,
    const INT ldafac,
    const INT* const restrict ipiv,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j, k;
    f32 anorm, eps;
    c64 t;
    INT minmn;

    if (m <= 0 || n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    anorm = clange("1", m, n, A, lda, rwork);

    minmn = (m < n) ? m : n;

    for (k = n - 1; k >= 0; k--) {
        if (k >= m) {
            cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        m, AFAC, ldafac, &AFAC[k * ldafac], 1);
        } else {
            t = AFAC[k + k * ldafac];
            if (k + 1 < m) {
                cblas_cscal(m - k - 1, &t, &AFAC[k + 1 + k * ldafac], 1);
                if (k > 0) {
                    cblas_cgemv(CblasColMajor, CblasNoTrans, m - k - 1, k, &CONE,
                                &AFAC[k + 1], ldafac, &AFAC[k * ldafac], 1, &CONE,
                                &AFAC[k + 1 + k * ldafac], 1);
                }
            }

            if (k > 0) {
                c64 dot;
                cblas_cdotu_sub(k, &AFAC[k], ldafac, &AFAC[k * ldafac], 1, &dot);
                AFAC[k + k * ldafac] = t + dot;
            }

            if (k > 0) {
                cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    }

    claswp(n, AFAC, ldafac, 0, minmn - 1, ipiv, -1);

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
        }
    }

    *resid = clange("1", m, n, AFAC, ldafac, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f32)n) / anorm) / eps;
    }
}
