/**
 * @file sgesc2.c
 * @brief Solve a system using LU factorization with complete pivoting.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGESC2 solves a system of linear equations
 *
 *           A * X = scale * RHS
 *
 * with a general N-by-N matrix A using the LU factorization with
 * complete pivoting computed by SGETC2.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     A     The LU part of the factorization of the n-by-n
 *                      matrix A computed by SGETC2: A = P * L * U * Q
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in,out] rhs   On entry, the right hand side vector b.
 *                      On exit, the solution vector X.
 *                      Array of dimension n.
 * @param[in]     ipiv  The pivot indices; for 0 <= i < n, row i of the
 *                      matrix has been interchanged with row ipiv[i].
 *                      Array of dimension n, 0-based.
 * @param[in]     jpiv  The pivot indices; for 0 <= j < n, column j of the
 *                      matrix has been interchanged with column jpiv[j].
 *                      Array of dimension n, 0-based.
 * @param[out]    scale On exit, SCALE contains the scale factor. SCALE is chosen
 *                      0 <= SCALE <= 1 to prevent overflow in the solution.
 */
void sgesc2(
    const INT n,
    const f32* restrict A,
    const INT lda,
    f32* restrict rhs,
    const INT* restrict ipiv,
    const INT* restrict jpiv,
    f32* scale)
{
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    f32 eps, smlnum, bignum, temp;
    INT i, j, imax;

    // Quick return if possible
    if (n == 0) {
        *scale = ONE;
        return;
    }

    // Set constant to control overflow
    eps = slamch("P");
    smlnum = slamch("S") / eps;
    bignum = ONE / smlnum;
    (void)bignum;  /* Computed but unused in LAPACK reference */

    // Apply permutations IPIV to RHS (forward direction)
    // Apply swaps for indices 0 to n-2
    for (i = 0; i < n - 1; i++) {
        if (ipiv[i] != i) {
            temp = rhs[i];
            rhs[i] = rhs[ipiv[i]];
            rhs[ipiv[i]] = temp;
        }
    }

    // Solve for L part (forward substitution)
    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            rhs[j] = rhs[j] - A[j + i * lda] * rhs[i];
        }
    }

    // Solve for U part (backward substitution)
    *scale = ONE;

    // Check for scaling to prevent overflow
    imax = cblas_isamax(n, rhs, 1);
    if (TWO * smlnum * fabsf(rhs[imax]) > fabsf(A[n - 1 + (n - 1) * lda])) {
        temp = (ONE / TWO) / fabsf(rhs[imax]);
        cblas_sscal(n, temp, rhs, 1);
        *scale = (*scale) * temp;
    }

    // Backward substitution for U
    for (i = n - 1; i >= 0; i--) {
        temp = ONE / A[i + i * lda];
        rhs[i] = rhs[i] * temp;
        for (j = i + 1; j < n; j++) {
            rhs[i] = rhs[i] - rhs[j] * (A[i + j * lda] * temp);
        }
    }

    // Apply permutations JPIV to the solution (RHS) in reverse direction
    for (i = n - 2; i >= 0; i--) {
        if (jpiv[i] != i) {
            temp = rhs[i];
            rhs[i] = rhs[jpiv[i]];
            rhs[jpiv[i]] = temp;
        }
    }
}
