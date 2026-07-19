/**
 * @file dgesc2.c
 * @brief Solve a system using LU factorization with complete pivoting.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DGESC2 solves a system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = scale * RHS
 * @endrst
 * with a general `n`-by-`n` matrix `A` using the LU factorization with
 * complete pivoting computed by DGETC2.
 *
 * @param[in]     n     The number of columns of the matrix `A`.
 * @param[in]     A     Array of dimension (`lda`, `n`).
 *                      On entry, the LU part of the factorization of the n-by-n
 *                      matrix `A` computed by DGETC2: A = P * L * U * Q
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[in,out] rhs   Array of dimension `n`.
 *                      On entry, the right hand side vector b.
 *                      On exit, the solution vector X.
 * @param[in]     ipiv  Array of dimension `n`.
 *                      The pivot indices; for `0<=i<n`, row i of the
 *                      matrix has been interchanged with row `ipiv[i]`.
 * @param[in]     jpiv  Array of dimension `n`.
 *                      The pivot indices; for `0<=j<n`, column j of the
 *                      matrix has been interchanged with column `jpiv[j]`.
 * @param[out]    scale On exit, `scale` contains the scale factor. `scale` is chosen
 *                      `0<=scale<=1` to prevent overflow in the solution.
 */
void dgesc2(
    const INT n,
    const f64* restrict A,
    const INT lda,
    f64* restrict rhs,
    const INT* restrict ipiv,
    const INT* restrict jpiv,
    f64* scale)
{
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 eps, smlnum, bignum, temp;
    INT i, j, imax;

    // Quick return if possible
    if (n == 0) {
        *scale = ONE;
        return;
    }

    // Set constant to control overflow
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
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
    imax = cblas_idamax(n, rhs, 1);
    if (TWO * smlnum * fabs(rhs[imax]) > fabs(A[n - 1 + (n - 1) * lda])) {
        temp = (ONE / TWO) / fabs(rhs[imax]);
        cblas_dscal(n, temp, rhs, 1);
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
