/**
 * @file dlagge.c
 * @brief DLAGGE generates a real general m by n matrix A, by pre- and post-
 *        multiplying a real diagonal matrix D with random orthogonal matrices.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlagge.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declaration */
extern void xerbla(const char* srname, const int info);

/**
 * DLAGGE generates a real general m by n matrix A, by pre- and post-
 * multiplying a real diagonal matrix D with random orthogonal matrices:
 * A = U*D*V. The lower and upper bandwidths may then be reduced to
 * kl and ku by additional orthogonal transformations.
 *
 * @param[in] m
 *     The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *     The number of columns of the matrix A. n >= 0.
 *
 * @param[in] kl
 *     The number of nonzero subdiagonals within the band of A.
 *     0 <= kl <= m-1.
 *
 * @param[in] ku
 *     The number of nonzero superdiagonals within the band of A.
 *     0 <= ku <= n-1.
 *
 * @param[in] d
 *     The diagonal elements of the diagonal matrix D.
 *     Dimension: min(m, n). 0-based indexing.
 *
 * @param[out] A
 *     The generated m by n matrix A. Dimension (lda, n). 0-based indexing.
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in] seed
 *     Random number seed for generating orthogonal transformations.
 *
 * @param[out] work
 *     Workspace array of dimension (m + n).
 *
 * @param[out] info
 *     = 0: successful exit
 *     < 0: if info = -i, the i-th argument had an illegal value
 */
void dlagge(
    const int m,
    const int n,
    const int kl,
    const int ku,
    const double* d,
    double* A,
    const int lda,
    uint64_t seed,
    double* work,
    int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j;
    double tau, wa, wb, wn;
    int minmn = (m < n) ? m : n;

    /* Initialize RNG with seed */
    rng_seed(seed);

    /* Test the input arguments */
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
        xerbla("DLAGGE", -(*info));
        return;
    }

    /* Initialize A to diagonal matrix */
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            A[i + j * lda] = ZERO;
        }
    }
    for (i = 0; i < minmn; i++) {
        A[i + i * lda] = d[i];
    }

    /* Quick exit if the user wants a diagonal matrix */
    if (kl == 0 && ku == 0) {
        return;
    }

    /* Pre- and post-multiply A by random orthogonal matrices */
    for (i = minmn - 1; i >= 0; i--) {
        if (i < m - 1) {
            /* Generate random reflection */
            int len = m - i;
            for (j = 0; j < len; j++) {
                work[j] = rng_normal();
            }
            wn = cblas_dnrm2(len, work, 1);
            wa = (work[0] >= 0.0) ? wn : -wn;
            if (wn == ZERO) {
                tau = ZERO;
            } else {
                wb = work[0] + wa;
                cblas_dscal(len - 1, ONE / wb, &work[1], 1);
                work[0] = ONE;
                tau = wb / wa;
            }

            /* Multiply A(i:m-1, i:n-1) by random reflection from the left */
            /* w = A' * v, then A = A - tau * v * w' */
            cblas_dgemv(CblasColMajor, CblasTrans, len, n - i, ONE,
                        &A[i + i * lda], lda, work, 1, ZERO, &work[m], 1);
            cblas_dger(CblasColMajor, len, n - i, -tau, work, 1,
                       &work[m], 1, &A[i + i * lda], lda);
        }
        if (i < n - 1) {
            /* Generate random reflection */
            int len = n - i;
            for (j = 0; j < len; j++) {
                work[j] = rng_normal();
            }
            wn = cblas_dnrm2(len, work, 1);
            wa = (work[0] >= 0.0) ? wn : -wn;
            if (wn == ZERO) {
                tau = ZERO;
            } else {
                wb = work[0] + wa;
                cblas_dscal(len - 1, ONE / wb, &work[1], 1);
                work[0] = ONE;
                tau = wb / wa;
            }

            /* Multiply A(i:m-1, i:n-1) by random reflection from the right */
            /* w = A * v, then A = A - tau * w * v' */
            cblas_dgemv(CblasColMajor, CblasNoTrans, m - i, len, ONE,
                        &A[i + i * lda], lda, work, 1, ZERO, &work[n], 1);
            cblas_dger(CblasColMajor, m - i, len, -tau, &work[n], 1,
                       work, 1, &A[i + i * lda], lda);
        }
    }

    /* Reduce number of subdiagonals to kl and number of superdiagonals to ku */
    int maxiter = (m - 1 - kl > n - 1 - ku) ? m - 1 - kl : n - 1 - ku;
    for (i = 0; i < maxiter; i++) {
        if (kl <= ku) {
            /* Annihilate subdiagonal elements first (necessary if kl = 0) */
            if (i < m - 1 - kl && i < n) {
                /* Generate reflection to annihilate A(kl+i+1:m-1, i) */
                int len = m - kl - i;
                wn = cblas_dnrm2(len, &A[kl + i + i * lda], 1);
                wa = (A[kl + i + i * lda] >= 0.0) ? wn : -wn;
                if (wn == ZERO) {
                    tau = ZERO;
                } else {
                    wb = A[kl + i + i * lda] + wa;
                    cblas_dscal(len - 1, ONE / wb, &A[kl + i + 1 + i * lda], 1);
                    A[kl + i + i * lda] = ONE;
                    tau = wb / wa;
                }

                /* Apply reflection to A(kl+i:m-1, i+1:n-1) from the left */
                if (n - i - 1 > 0) {
                    cblas_dgemv(CblasColMajor, CblasTrans, len, n - i - 1, ONE,
                                &A[kl + i + (i + 1) * lda], lda,
                                &A[kl + i + i * lda], 1, ZERO, work, 1);
                    cblas_dger(CblasColMajor, len, n - i - 1, -tau,
                               &A[kl + i + i * lda], 1, work, 1,
                               &A[kl + i + (i + 1) * lda], lda);
                }
                A[kl + i + i * lda] = -wa;
            }

            if (i < n - 1 - ku && i < m) {
                /* Generate reflection to annihilate A(i, ku+i+1:n-1) */
                int len = n - ku - i;
                wn = cblas_dnrm2(len, &A[i + (ku + i) * lda], lda);
                wa = (A[i + (ku + i) * lda] >= 0.0) ? wn : -wn;
                if (wn == ZERO) {
                    tau = ZERO;
                } else {
                    wb = A[i + (ku + i) * lda] + wa;
                    cblas_dscal(len - 1, ONE / wb, &A[i + (ku + i + 1) * lda], lda);
                    A[i + (ku + i) * lda] = ONE;
                    tau = wb / wa;
                }

                /* Apply reflection to A(i+1:m-1, ku+i:n-1) from the right */
                if (m - i - 1 > 0) {
                    cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, len, ONE,
                                &A[i + 1 + (ku + i) * lda], lda,
                                &A[i + (ku + i) * lda], lda, ZERO, work, 1);
                    cblas_dger(CblasColMajor, m - i - 1, len, -tau, work, 1,
                               &A[i + (ku + i) * lda], lda,
                               &A[i + 1 + (ku + i) * lda], lda);
                }
                A[i + (ku + i) * lda] = -wa;
            }
        } else {
            /* Annihilate superdiagonal elements first (necessary if ku = 0) */
            if (i < n - 1 - ku && i < m) {
                /* Generate reflection to annihilate A(i, ku+i+1:n-1) */
                int len = n - ku - i;
                wn = cblas_dnrm2(len, &A[i + (ku + i) * lda], lda);
                wa = (A[i + (ku + i) * lda] >= 0.0) ? wn : -wn;
                if (wn == ZERO) {
                    tau = ZERO;
                } else {
                    wb = A[i + (ku + i) * lda] + wa;
                    cblas_dscal(len - 1, ONE / wb, &A[i + (ku + i + 1) * lda], lda);
                    A[i + (ku + i) * lda] = ONE;
                    tau = wb / wa;
                }

                /* Apply reflection to A(i+1:m-1, ku+i:n-1) from the right */
                if (m - i - 1 > 0) {
                    cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, len, ONE,
                                &A[i + 1 + (ku + i) * lda], lda,
                                &A[i + (ku + i) * lda], lda, ZERO, work, 1);
                    cblas_dger(CblasColMajor, m - i - 1, len, -tau, work, 1,
                               &A[i + (ku + i) * lda], lda,
                               &A[i + 1 + (ku + i) * lda], lda);
                }
                A[i + (ku + i) * lda] = -wa;
            }

            if (i < m - 1 - kl && i < n) {
                /* Generate reflection to annihilate A(kl+i+1:m-1, i) */
                int len = m - kl - i;
                wn = cblas_dnrm2(len, &A[kl + i + i * lda], 1);
                wa = (A[kl + i + i * lda] >= 0.0) ? wn : -wn;
                if (wn == ZERO) {
                    tau = ZERO;
                } else {
                    wb = A[kl + i + i * lda] + wa;
                    cblas_dscal(len - 1, ONE / wb, &A[kl + i + 1 + i * lda], 1);
                    A[kl + i + i * lda] = ONE;
                    tau = wb / wa;
                }

                /* Apply reflection to A(kl+i:m-1, i+1:n-1) from the left */
                if (n - i - 1 > 0) {
                    cblas_dgemv(CblasColMajor, CblasTrans, len, n - i - 1, ONE,
                                &A[kl + i + (i + 1) * lda], lda,
                                &A[kl + i + i * lda], 1, ZERO, work, 1);
                    cblas_dger(CblasColMajor, len, n - i - 1, -tau,
                               &A[kl + i + i * lda], 1, work, 1,
                               &A[kl + i + (i + 1) * lda], lda);
                }
                A[kl + i + i * lda] = -wa;
            }
        }

        /* Zero out elements outside the band */
        if (i < n) {
            for (j = kl + i + 1; j < m; j++) {
                A[j + i * lda] = ZERO;
            }
        }
        if (i < m) {
            for (j = ku + i + 1; j < n; j++) {
                A[i + j * lda] = ZERO;
            }
        }
    }
}
