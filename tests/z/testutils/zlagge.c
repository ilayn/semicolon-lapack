/**
 * @file zlagge.c
 * @brief ZLAGGE generates a complex general m by n matrix A, by pre- and post-
 *        multiplying a real diagonal matrix D with random unitary matrices.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlagge.f
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * ZLAGGE generates a complex general m by n matrix A, by pre- and post-
 * multiplying a real diagonal matrix D with random unitary matrices:
 * A = U*D*V. The lower and upper bandwidths may then be reduced to
 * kl and ku by additional unitary transformations.
 *
 * @param[in] m       The number of rows of the matrix A. m >= 0.
 * @param[in] n       The number of columns of the matrix A. n >= 0.
 * @param[in] kl      The number of nonzero subdiagonals. 0 <= kl <= m-1.
 * @param[in] ku      The number of nonzero superdiagonals. 0 <= ku <= n-1.
 * @param[in] d       Real diagonal elements. Dimension: min(m, n).
 * @param[out] A      Complex array, dimension (lda, n).
 * @param[in] lda     The leading dimension of A. lda >= max(1, m).
 * @param[out] work   Complex workspace of dimension (m + n).
 * @param[out] info   = 0: successful exit.
 * @param[in,out] state  RNG state array.
 */
void zlagge(
    const INT m, const INT n,
    const INT kl, const INT ku,
    const f64* d,
    c128* A, const INT lda,
    c128* work, INT* info,
    uint64_t state[static 4])
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, j;
    f64 wn, tau;
    c128 wa, wb;
    INT minmn = (m < n) ? m : n;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0 || kl > m - 1) {
        *info = -3;
    } else if (ku < 0 || ku > n - 1) {
        *info = -4;
    } else if (lda < ((m > 1) ? m : 1)) {
        *info = -7;
    }
    if (*info < 0) {
        xerbla("ZLAGGE", -(*info));
        return;
    }

    /* Initialize A to diagonal matrix */
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            A[i + j * lda] = CZERO;
        }
    }
    for (i = 0; i < minmn; i++) {
        A[i + i * lda] = CMPLX(d[i], 0.0);
    }

    if (kl == 0 && ku == 0) {
        return;
    }

    /* Pre- and post-multiply A by random unitary matrices */
    for (i = minmn - 1; i >= 0; i--) {
        if (i < m - 1) {
            INT len = m - i;
            zlarnv_rng(3, len, work, state);
            wn = cblas_dznrm2(len, work, 1);
            wa = (wn / cabs(work[0])) * work[0];
            if (wn == 0.0) {
                tau = 0.0;
            } else {
                wb = work[0] + wa;
                c128 scale = CONE / wb;
                cblas_zscal(len - 1, &scale, &work[1], 1);
                work[0] = CONE;
                tau = creal(wb / wa);
            }

            c128 neg_tau = CMPLX(-tau, 0.0);
            cblas_zgemv(CblasColMajor, CblasConjTrans, len, n - i, &CONE,
                        &A[i + i * lda], lda, work, 1, &CZERO, &work[m], 1);
            cblas_zgerc(CblasColMajor, len, n - i, &neg_tau, work, 1,
                        &work[m], 1, &A[i + i * lda], lda);
        }
        if (i < n - 1) {
            INT len = n - i;
            zlarnv_rng(3, len, work, state);
            wn = cblas_dznrm2(len, work, 1);
            wa = (wn / cabs(work[0])) * work[0];
            if (wn == 0.0) {
                tau = 0.0;
            } else {
                wb = work[0] + wa;
                c128 scale = CONE / wb;
                cblas_zscal(len - 1, &scale, &work[1], 1);
                work[0] = CONE;
                tau = creal(wb / wa);
            }

            c128 neg_tau = CMPLX(-tau, 0.0);
            cblas_zgemv(CblasColMajor, CblasNoTrans, m - i, len, &CONE,
                        &A[i + i * lda], lda, work, 1, &CZERO, &work[n], 1);
            cblas_zgerc(CblasColMajor, m - i, len, &neg_tau, &work[n], 1,
                        work, 1, &A[i + i * lda], lda);
        }
    }

    /* Reduce number of subdiagonals to kl and number of superdiagonals to ku */
    INT maxiter = (m - 1 - kl > n - 1 - ku) ? m - 1 - kl : n - 1 - ku;
    for (i = 0; i < maxiter; i++) {
        if (kl <= ku) {
            if (i < m - 1 - kl && i < n) {
                INT len = m - kl - i;
                wn = cblas_dznrm2(len, &A[kl + i + i * lda], 1);
                wa = (wn / cabs(A[kl + i + i * lda])) * A[kl + i + i * lda];
                if (wn == 0.0) {
                    tau = 0.0;
                } else {
                    wb = A[kl + i + i * lda] + wa;
                    c128 scale = CONE / wb;
                    cblas_zscal(len - 1, &scale, &A[kl + i + 1 + i * lda], 1);
                    A[kl + i + i * lda] = CONE;
                    tau = creal(wb / wa);
                }

                if (n - i - 1 > 0) {
                    c128 neg_tau = CMPLX(-tau, 0.0);
                    cblas_zgemv(CblasColMajor, CblasConjTrans, len, n - i - 1,
                                &CONE, &A[kl + i + (i + 1) * lda], lda,
                                &A[kl + i + i * lda], 1, &CZERO, work, 1);
                    cblas_zgerc(CblasColMajor, len, n - i - 1, &neg_tau,
                                &A[kl + i + i * lda], 1, work, 1,
                                &A[kl + i + (i + 1) * lda], lda);
                }
                A[kl + i + i * lda] = -wa;
            }

            if (i < n - 1 - ku && i < m) {
                INT len = n - ku - i;
                wn = cblas_dznrm2(len, &A[i + (ku + i) * lda], lda);
                wa = (wn / cabs(A[i + (ku + i) * lda])) * A[i + (ku + i) * lda];
                if (wn == 0.0) {
                    tau = 0.0;
                } else {
                    wb = A[i + (ku + i) * lda] + wa;
                    c128 scale = CONE / wb;
                    cblas_zscal(len - 1, &scale, &A[i + (ku + i + 1) * lda], lda);
                    A[i + (ku + i) * lda] = CONE;
                    tau = creal(wb / wa);
                }

                zlacgv(len, &A[i + (ku + i) * lda], lda);
                if (m - i - 1 > 0) {
                    c128 neg_tau = CMPLX(-tau, 0.0);
                    cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, len,
                                &CONE, &A[i + 1 + (ku + i) * lda], lda,
                                &A[i + (ku + i) * lda], lda, &CZERO, work, 1);
                    cblas_zgerc(CblasColMajor, m - i - 1, len, &neg_tau,
                                work, 1, &A[i + (ku + i) * lda], lda,
                                &A[i + 1 + (ku + i) * lda], lda);
                }
                A[i + (ku + i) * lda] = -wa;
            }
        } else {
            if (i < n - 1 - ku && i < m) {
                INT len = n - ku - i;
                wn = cblas_dznrm2(len, &A[i + (ku + i) * lda], lda);
                wa = (wn / cabs(A[i + (ku + i) * lda])) * A[i + (ku + i) * lda];
                if (wn == 0.0) {
                    tau = 0.0;
                } else {
                    wb = A[i + (ku + i) * lda] + wa;
                    c128 scale = CONE / wb;
                    cblas_zscal(len - 1, &scale, &A[i + (ku + i + 1) * lda], lda);
                    A[i + (ku + i) * lda] = CONE;
                    tau = creal(wb / wa);
                }

                zlacgv(len, &A[i + (ku + i) * lda], lda);
                if (m - i - 1 > 0) {
                    c128 neg_tau = CMPLX(-tau, 0.0);
                    cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, len,
                                &CONE, &A[i + 1 + (ku + i) * lda], lda,
                                &A[i + (ku + i) * lda], lda, &CZERO, work, 1);
                    cblas_zgerc(CblasColMajor, m - i - 1, len, &neg_tau,
                                work, 1, &A[i + (ku + i) * lda], lda,
                                &A[i + 1 + (ku + i) * lda], lda);
                }
                A[i + (ku + i) * lda] = -wa;
            }

            if (i < m - 1 - kl && i < n) {
                INT len = m - kl - i;
                wn = cblas_dznrm2(len, &A[kl + i + i * lda], 1);
                wa = (wn / cabs(A[kl + i + i * lda])) * A[kl + i + i * lda];
                if (wn == 0.0) {
                    tau = 0.0;
                } else {
                    wb = A[kl + i + i * lda] + wa;
                    c128 scale = CONE / wb;
                    cblas_zscal(len - 1, &scale, &A[kl + i + 1 + i * lda], 1);
                    A[kl + i + i * lda] = CONE;
                    tau = creal(wb / wa);
                }

                if (n - i - 1 > 0) {
                    c128 neg_tau = CMPLX(-tau, 0.0);
                    cblas_zgemv(CblasColMajor, CblasConjTrans, len, n - i - 1,
                                &CONE, &A[kl + i + (i + 1) * lda], lda,
                                &A[kl + i + i * lda], 1, &CZERO, work, 1);
                    cblas_zgerc(CblasColMajor, len, n - i - 1, &neg_tau,
                                &A[kl + i + i * lda], 1, work, 1,
                                &A[kl + i + (i + 1) * lda], lda);
                }
                A[kl + i + i * lda] = -wa;
            }
        }

        if (i < n) {
            for (j = kl + i + 1; j < m; j++) {
                A[j + i * lda] = CZERO;
            }
        }
        if (i < m) {
            for (j = ku + i + 1; j < n; j++) {
                A[i + j * lda] = CZERO;
            }
        }
    }
}
