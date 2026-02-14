/**
 * @file sgetc2.c
 * @brief LU factorization with complete pivoting.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGETC2 computes an LU factorization with complete pivoting of the
 * n-by-n matrix A. The factorization has the form A = P * L * U * Q,
 * where P and Q are permutation matrices, L is lower triangular with
 * unit diagonal elements and U is upper triangular.
 *
 * This is the Level 2 BLAS algorithm.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     On entry, the n-by-n matrix A to be factored.
 *                      On exit, the factors L and U from the factorization
 *                      A = P*L*U*Q; the unit diagonal elements of L are not stored.
 *                      If U(k, k) appears to be less than SMIN, U(k, k) is given the
 *                      value of SMIN, i.e., giving a nonsingular perturbed system.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[out]    ipiv  The pivot indices; for 0 <= i < n, row i of the
 *                      matrix has been interchanged with row ipiv[i].
 *                      Array of dimension n, 0-based.
 * @param[out]    jpiv  The pivot indices; for 0 <= j < n, column j of the
 *                      matrix has been interchanged with column jpiv[j].
 *                      Array of dimension n, 0-based.
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - > 0: if info = k, U(k-1, k-1) is likely to produce overflow if
 *                           we try to solve for x in Ax = b. So U is perturbed to
 *                           avoid the overflow.
 */
void sgetc2(
    const int n,
    f32* restrict A,
    const int lda,
    int* restrict ipiv,
    int* restrict jpiv,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;

    f32 eps, smlnum, bignum, smin, xmax;
    int i, ip, ipv, j, jp, jpv;

    *info = 0;

    // Quick return if possible
    if (n == 0) {
        return;
    }

    // Set constants to control overflow
    eps = slamch("P");
    smlnum = slamch("S") / eps;
    bignum = ONE / smlnum;
    (void)bignum;  /* Computed but unused in LAPACK reference */

    // Handle the case n=1 by itself
    if (n == 1) {
        ipiv[0] = 0;
        jpiv[0] = 0;
        if (fabsf(A[0]) < smlnum) {
            *info = 1;
            A[0] = smlnum;
        }
        return;
    }

    // Factorize A using complete pivoting.
    // Set pivots less than SMIN to SMIN.
    smin = ZERO;  // Will be set on first iteration

    for (i = 0; i < n - 1; i++) {
        // Find max element in matrix A[i:n-1, i:n-1]
        xmax = ZERO;
        ipv = i;
        jpv = i;

        for (jp = i; jp < n; jp++) {
            for (ip = i; ip < n; ip++) {
                if (fabsf(A[ip + jp * lda]) >= xmax) {
                    xmax = fabsf(A[ip + jp * lda]);
                    ipv = ip;
                    jpv = jp;
                }
            }
        }

        // Set SMIN on first iteration
        if (i == 0) {
            smin = fmaxf(eps * xmax, smlnum);
        }

        // Swap rows
        if (ipv != i) {
            cblas_sswap(n, &A[ipv], lda, &A[i], lda);
        }
        ipiv[i] = ipv;

        // Swap columns
        if (jpv != i) {
            cblas_sswap(n, &A[jpv * lda], 1, &A[i * lda], 1);
        }
        jpiv[i] = jpv;

        // Check for singularity
        if (fabsf(A[i + i * lda]) < smin) {
            *info = i + 1;
            A[i + i * lda] = smin;
        }

        // Compute elements i+1:n-1 of column i (L factor)
        for (j = i + 1; j < n; j++) {
            A[j + i * lda] = A[j + i * lda] / A[i + i * lda];
        }

        // Update trailing submatrix
        // A[i+1:n-1, i+1:n-1] -= A[i+1:n-1, i] * A[i, i+1:n-1]
        cblas_sger(CblasColMajor, n - i - 1, n - i - 1, NEG_ONE,
                   &A[i + 1 + i * lda], 1,           // column vector
                   &A[i + (i + 1) * lda], lda,       // row vector
                   &A[i + 1 + (i + 1) * lda], lda);
    }

    // Check last pivot
    if (fabsf(A[n - 1 + (n - 1) * lda]) < smin) {
        *info = n;
        A[n - 1 + (n - 1) * lda] = smin;
    }

    // Set last pivots to n-1 (0-based)
    ipiv[n - 1] = n - 1;
    jpiv[n - 1] = n - 1;
}
