/**
 * @file dlagsy.c
 * @brief DLAGSY generates a real symmetric matrix A, by pre- and post-
 *        multiplying a real diagonal matrix D with a random orthogonal matrix.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlagsy.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations */
extern void xerbla(const char* srname, const int info);
extern double dlamch(const char* cmach);

/**
 * DLAGSY generates a real symmetric matrix A, by pre- and post-
 * multiplying a real diagonal matrix D with a random orthogonal matrix:
 * A = U*D*U'. The semi-bandwidth may then be reduced to k by additional
 * orthogonal transformations.
 *
 * @param[in] n
 *     The order of the matrix A. n >= 0.
 *
 * @param[in] k
 *     The number of nonzero subdiagonals within the band of A.
 *     0 <= k <= n-1.
 *
 * @param[in] d
 *     The diagonal elements of the diagonal matrix D.
 *     Dimension: n. 0-based indexing.
 *
 * @param[out] A
 *     The generated n by n symmetric matrix A (the full matrix is stored).
 *     Dimension (lda, n). 0-based indexing.
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= n.
 *
 * @param[out] work
 *     Workspace array of dimension (2*n).
 *
 * @param[out] info
 *     = 0: successful exit
 *     < 0: if info = -i, the i-th argument had an illegal value
 */
void dlagsy(
    const int n,
    const int k,
    const double* d,
    double* A,
    const int lda,
    double* work,
    int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double HALF = 0.5;

    int i, j;
    double alpha, tau, wa, wb, wn;
    double safmin = dlamch("S");

    /* Test the input arguments */
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (k < 0 || k > n - 1) {
        *info = -2;
    } else if (lda < ((n > 1) ? n : 1)) {
        *info = -5;
    }
    if (*info < 0) {
        xerbla("DLAGSY", -(*info));
        return;
    }

    /* Initialize lower triangle of A to diagonal matrix */
    for (j = 0; j < n; j++) {
        for (i = j + 1; i < n; i++) {
            A[i + j * lda] = ZERO;
        }
    }
    for (i = 0; i < n; i++) {
        A[i + i * lda] = d[i];
    }

    /* Generate lower triangle of symmetric matrix */
    for (i = n - 2; i >= 0; i--) {
        /* Generate random reflection */
        int len = n - i;
        for (j = 0; j < len; j++) {
            work[j] = rng_normal();
        }
        wn = cblas_dnrm2(len, work, 1);
        wa = (work[0] >= 0.0) ? wn : -wn;
        if (wn < safmin) {
            tau = ZERO;
        } else {
            wb = work[0] + wa;
            cblas_dscal(len - 1, ONE / wb, &work[1], 1);
            work[0] = ONE;
            tau = wb / wa;
        }

        /* Apply random reflection to A(i:n-1, i:n-1) from the left and the right */

        /* Compute y := tau * A * u */
        cblas_dsymv(CblasColMajor, CblasLower, len, tau,
                    &A[i + i * lda], lda, work, 1, ZERO, &work[n], 1);

        /* Compute v := y - 1/2 * tau * (y, u) * u */
        alpha = -HALF * tau * cblas_ddot(len, &work[n], 1, work, 1);
        cblas_daxpy(len, alpha, work, 1, &work[n], 1);

        /* Apply the transformation as a rank-2 update to A(i:n-1, i:n-1) */
        cblas_dsyr2(CblasColMajor, CblasLower, len, -ONE, work, 1,
                    &work[n], 1, &A[i + i * lda], lda);
    }

    /* Reduce number of subdiagonals to k */
    for (i = 0; i < n - 1 - k; i++) {
        /* Generate reflection to annihilate A(k+i+1:n-1, i) */
        int len = n - k - i;
        wn = cblas_dnrm2(len, &A[k + i + i * lda], 1);
        wa = (A[k + i + i * lda] >= 0.0) ? wn : -wn;
        if (wn < safmin) {
            tau = ZERO;
        } else {
            wb = A[k + i + i * lda] + wa;
            cblas_dscal(len - 1, ONE / wb, &A[k + i + 1 + i * lda], 1);
            A[k + i + i * lda] = ONE;
            tau = wb / wa;
        }

        /* Apply reflection to A(k+i:n-1, i+1:k+i-1) from the left */
        if (k - 1 > 0) {
            cblas_dgemv(CblasColMajor, CblasTrans, len, k - 1, ONE,
                        &A[k + i + (i + 1) * lda], lda,
                        &A[k + i + i * lda], 1, ZERO, work, 1);
            cblas_dger(CblasColMajor, len, k - 1, -tau,
                       &A[k + i + i * lda], 1, work, 1,
                       &A[k + i + (i + 1) * lda], lda);
        }

        /* Apply reflection to A(k+i:n-1, k+i:n-1) from the left and the right */

        /* Compute y := tau * A * u */
        cblas_dsymv(CblasColMajor, CblasLower, len, tau,
                    &A[k + i + (k + i) * lda], lda,
                    &A[k + i + i * lda], 1, ZERO, work, 1);

        /* Compute v := y - 1/2 * tau * (y, u) * u */
        alpha = -HALF * tau * cblas_ddot(len, work, 1, &A[k + i + i * lda], 1);
        cblas_daxpy(len, alpha, &A[k + i + i * lda], 1, work, 1);

        /* Apply symmetric rank-2 update to A(k+i:n-1, k+i:n-1) */
        cblas_dsyr2(CblasColMajor, CblasLower, len, -ONE,
                    &A[k + i + i * lda], 1, work, 1,
                    &A[k + i + (k + i) * lda], lda);

        A[k + i + i * lda] = -wa;
        for (j = k + i + 1; j < n; j++) {
            A[j + i * lda] = ZERO;
        }
    }

    /* Store full symmetric matrix */
    for (j = 0; j < n; j++) {
        for (i = j + 1; i < n; i++) {
            A[j + i * lda] = A[i + j * lda];
        }
    }
}
