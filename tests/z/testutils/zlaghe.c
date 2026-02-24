/**
 * @file zlaghe.c
 * @brief ZLAGHE generates a complex Hermitian matrix A, by pre- and post-
 *        multiplying a real diagonal matrix D with a random unitary matrix.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlaghe.f
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * ZLAGHE generates a complex Hermitian matrix A, by pre- and post-
 * multiplying a real diagonal matrix D with a random unitary matrix:
 * A = U*D*U'. The semi-bandwidth may then be reduced to k by additional
 * unitary transformations.
 *
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] k       The number of nonzero subdiagonals within the band of A.
 *                     0 <= k <= n-1.
 * @param[in] d       Real diagonal elements. Dimension: n.
 * @param[out] A      Complex array, dimension (lda, n).
 * @param[in] lda     The leading dimension of A. lda >= max(1, n).
 * @param[out] work   Complex workspace of dimension (2*n).
 * @param[out] info   = 0: successful exit.
 * @param[in,out] state  RNG state array.
 */
void zlaghe(
    const INT n, const INT k,
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

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (k < 0 || k > n - 1) {
        *info = -2;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -5;
    }
    if (*info < 0) {
        xerbla("ZLAGHE", -(*info));
        return;
    }

    /* Initialize lower triangle of A to diagonal matrix */
    for (j = 0; j < n; j++) {
        for (i = j + 1; i < n; i++) {
            A[i + j * lda] = CZERO;
        }
    }
    for (i = 0; i < n; i++) {
        A[i + i * lda] = CMPLX(d[i], 0.0);
    }

    /* Generate lower triangle of Hermitian matrix */
    for (i = n - 2; i >= 0; i--) {
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

        /* Apply random reflection to A(i:n-1,i:n-1) from the left
         * and the right */

        /* Compute y := tau * A * u */
        c128 tau_c = CMPLX(tau, 0.0);
        cblas_zhemv(CblasColMajor, CblasLower, len, &tau_c,
                    &A[i + i * lda], lda, work, 1, &CZERO, &work[n], 1);

        /* Compute v := y - 1/2 * tau * ( y, u ) * u */
        c128 dot;
        cblas_zdotc_sub(len, &work[n], 1, work, 1, &dot);
        c128 alpha = CMPLX(-0.5 * tau, 0.0) * dot;
        cblas_zaxpy(len, &alpha, work, 1, &work[n], 1);

        /* Apply the transformation as a rank-2 update to A(i:n-1,i:n-1) */
        c128 neg_one = CMPLX(-1.0, 0.0);
        cblas_zher2(CblasColMajor, CblasLower, len, &neg_one, work, 1,
                    &work[n], 1, &A[i + i * lda], lda);
    }

    /* Reduce number of subdiagonals to k */
    for (i = 0; i < n - 1 - k; i++) {
        INT len = n - k - i;
        wn = cblas_dznrm2(len, &A[k + i + i * lda], 1);
        wa = (wn / cabs(A[k + i + i * lda])) * A[k + i + i * lda];
        if (wn == 0.0) {
            tau = 0.0;
        } else {
            wb = A[k + i + i * lda] + wa;
            c128 scale = CONE / wb;
            cblas_zscal(len - 1, &scale, &A[k + i + 1 + i * lda], 1);
            A[k + i + i * lda] = CONE;
            tau = creal(wb / wa);
        }

        /* Apply reflection to A(k+i:n-1,i+1:k+i-1) from the left */
        if (k - 1 > 0) {
            c128 neg_tau = CMPLX(-tau, 0.0);
            cblas_zgemv(CblasColMajor, CblasConjTrans, len, k - 1,
                        &CONE, &A[k + i + (i + 1) * lda], lda,
                        &A[k + i + i * lda], 1, &CZERO, work, 1);
            cblas_zgerc(CblasColMajor, len, k - 1, &neg_tau,
                        &A[k + i + i * lda], 1, work, 1,
                        &A[k + i + (i + 1) * lda], lda);
        }

        /* Apply reflection to A(k+i:n-1,k+i:n-1) from the left and the right */

        /* Compute y := tau * A * u */
        c128 tau_c = CMPLX(tau, 0.0);
        cblas_zhemv(CblasColMajor, CblasLower, len, &tau_c,
                    &A[k + i + (k + i) * lda], lda,
                    &A[k + i + i * lda], 1, &CZERO, work, 1);

        /* Compute v := y - 1/2 * tau * ( y, u ) * u */
        c128 dot;
        cblas_zdotc_sub(len, work, 1, &A[k + i + i * lda], 1, &dot);
        c128 alpha = CMPLX(-0.5 * tau, 0.0) * dot;
        cblas_zaxpy(len, &alpha, &A[k + i + i * lda], 1, work, 1);

        /* Apply Hermitian rank-2 update to A(k+i:n-1,k+i:n-1) */
        c128 neg_one = CMPLX(-1.0, 0.0);
        cblas_zher2(CblasColMajor, CblasLower, len, &neg_one,
                    &A[k + i + i * lda], 1, work, 1,
                    &A[k + i + (k + i) * lda], lda);

        A[k + i + i * lda] = -wa;
        for (j = k + i + 1; j < n; j++) {
            A[j + i * lda] = CZERO;
        }
    }

    /* Store full Hermitian matrix */
    for (j = 0; j < n; j++) {
        for (i = j + 1; i < n; i++) {
            A[j + i * lda] = conj(A[i + j * lda]);
        }
    }
}
