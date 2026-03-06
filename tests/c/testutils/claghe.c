/**
 * @file claghe.c
 * @brief CLAGHE generates a complex Hermitian matrix A, by pre- and post-
 *        multiplying a real diagonal matrix D with a random unitary matrix.
 *
 * Faithful port of LAPACK TESTING/MATGEN/claghe.f
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * CLAGHE generates a complex Hermitian matrix A, by pre- and post-
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
void claghe(
    const INT n, const INT k,
    const f32* d,
    c64* A, const INT lda,
    c64* work, INT* info,
    uint64_t state[static 4])
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i, j;
    f32 wn, tau;
    c64 wa, wb;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (k < 0 || k > n - 1) {
        *info = -2;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -5;
    }
    if (*info < 0) {
        xerbla("CLAGHE", -(*info));
        return;
    }

    /* Initialize lower triangle of A to diagonal matrix */
    for (j = 0; j < n; j++) {
        for (i = j + 1; i < n; i++) {
            A[i + j * lda] = CZERO;
        }
    }
    for (i = 0; i < n; i++) {
        A[i + i * lda] = CMPLXF(d[i], 0.0f);
    }

    /* Generate lower triangle of Hermitian matrix */
    for (i = n - 2; i >= 0; i--) {
        INT len = n - i;
        clarnv_rng(3, len, work, state);
        wn = cblas_scnrm2(len, work, 1);
        wa = (wn / cabsf(work[0])) * work[0];
        if (wn == 0.0f) {
            tau = 0.0f;
        } else {
            wb = work[0] + wa;
            c64 scale = CONE / wb;
            cblas_cscal(len - 1, &scale, &work[1], 1);
            work[0] = CONE;
            tau = crealf(wb / wa);
        }

        /* Apply random reflection to A(i:n-1,i:n-1) from the left
         * and the right */

        /* Compute y := tau * A * u */
        c64 tau_c = CMPLXF(tau, 0.0f);
        cblas_chemv(CblasColMajor, CblasLower, len, &tau_c,
                    &A[i + i * lda], lda, work, 1, &CZERO, &work[n], 1);

        /* Compute v := y - 1/2 * tau * ( y, u ) * u */
        c64 dot;
        cblas_cdotc_sub(len, &work[n], 1, work, 1, &dot);
        c64 alpha = CMPLXF(-0.5f * tau, 0.0f) * dot;
        cblas_caxpy(len, &alpha, work, 1, &work[n], 1);

        /* Apply the transformation as a rank-2 update to A(i:n-1,i:n-1) */
        c64 neg_one = CMPLXF(-1.0f, 0.0f);
        cblas_cher2(CblasColMajor, CblasLower, len, &neg_one, work, 1,
                    &work[n], 1, &A[i + i * lda], lda);
    }

    /* Reduce number of subdiagonals to k */
    for (i = 0; i < n - 1 - k; i++) {
        INT len = n - k - i;
        wn = cblas_scnrm2(len, &A[k + i + i * lda], 1);
        wa = (wn / cabsf(A[k + i + i * lda])) * A[k + i + i * lda];
        if (wn == 0.0f) {
            tau = 0.0f;
        } else {
            wb = A[k + i + i * lda] + wa;
            c64 scale = CONE / wb;
            cblas_cscal(len - 1, &scale, &A[k + i + 1 + i * lda], 1);
            A[k + i + i * lda] = CONE;
            tau = crealf(wb / wa);
        }

        /* Apply reflection to A(k+i:n-1,i+1:k+i-1) from the left */
        if (k - 1 > 0) {
            c64 neg_tau = CMPLXF(-tau, 0.0f);
            cblas_cgemv(CblasColMajor, CblasConjTrans, len, k - 1,
                        &CONE, &A[k + i + (i + 1) * lda], lda,
                        &A[k + i + i * lda], 1, &CZERO, work, 1);
            cblas_cgerc(CblasColMajor, len, k - 1, &neg_tau,
                        &A[k + i + i * lda], 1, work, 1,
                        &A[k + i + (i + 1) * lda], lda);
        }

        /* Apply reflection to A(k+i:n-1,k+i:n-1) from the left and the right */

        /* Compute y := tau * A * u */
        c64 tau_c = CMPLXF(tau, 0.0f);
        cblas_chemv(CblasColMajor, CblasLower, len, &tau_c,
                    &A[k + i + (k + i) * lda], lda,
                    &A[k + i + i * lda], 1, &CZERO, work, 1);

        /* Compute v := y - 1/2 * tau * ( y, u ) * u */
        c64 dot;
        cblas_cdotc_sub(len, work, 1, &A[k + i + i * lda], 1, &dot);
        c64 alpha = CMPLXF(-0.5f * tau, 0.0f) * dot;
        cblas_caxpy(len, &alpha, &A[k + i + i * lda], 1, work, 1);

        /* Apply Hermitian rank-2 update to A(k+i:n-1,k+i:n-1) */
        c64 neg_one = CMPLXF(-1.0f, 0.0f);
        cblas_cher2(CblasColMajor, CblasLower, len, &neg_one,
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
            A[j + i * lda] = conjf(A[i + j * lda]);
        }
    }
}
