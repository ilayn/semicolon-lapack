/**
 * @file zget01.c
 * @brief ZGET01 reconstructs a matrix A from its L*U factorization and
 *        computes the residual.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zget01(
    const INT m,
    const INT n,
    const c128* const restrict A,
    const INT lda,
    c128* const restrict AFAC,
    const INT ldafac,
    const INT* const restrict ipiv,
    f64* const restrict rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, j, k;
    f64 anorm, eps;
    c128 t;
    INT minmn;

    if (m <= 0 || n <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = zlange("1", m, n, A, lda, rwork);

    minmn = (m < n) ? m : n;

    for (k = n - 1; k >= 0; k--) {
        if (k >= m) {
            cblas_ztrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        m, AFAC, ldafac, &AFAC[k * ldafac], 1);
        } else {
            t = AFAC[k + k * ldafac];
            if (k + 1 < m) {
                cblas_zscal(m - k - 1, &t, &AFAC[k + 1 + k * ldafac], 1);
                if (k > 0) {
                    cblas_zgemv(CblasColMajor, CblasNoTrans, m - k - 1, k, &CONE,
                                &AFAC[k + 1], ldafac, &AFAC[k * ldafac], 1, &CONE,
                                &AFAC[k + 1 + k * ldafac], 1);
                }
            }

            if (k > 0) {
                c128 dot;
                cblas_zdotu_sub(k, &AFAC[k], ldafac, &AFAC[k * ldafac], 1, &dot);
                AFAC[k + k * ldafac] = t + dot;
            }

            if (k > 0) {
                cblas_ztrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                            k, AFAC, ldafac, &AFAC[k * ldafac], 1);
            }
        }
    }

    zlaswp(n, AFAC, ldafac, 0, minmn - 1, ipiv, -1);

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            AFAC[i + j * ldafac] = AFAC[i + j * ldafac] - A[i + j * lda];
        }
    }

    *resid = zlange("1", m, n, AFAC, ldafac, rwork);

    if (anorm <= ZERO) {
        if (*resid != ZERO) {
            *resid = ONE / eps;
        }
    } else {
        *resid = ((*resid / (f64)n) / anorm) / eps;
    }
}
