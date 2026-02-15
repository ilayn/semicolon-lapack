/**
 * @file cgesc2.c
 * @brief CGESC2 solves a system of linear equations using the LU factorization
 *        with complete pivoting computed by CGETC2.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGESC2 solves a system of linear equations
 *
 *           A * X = scale * RHS
 *
 * with a general N-by-N matrix A using the LU factorization with
 * complete pivoting computed by CGETC2.
 *
 * @param[in]     n     The number of columns of the matrix A.
 * @param[in]     A     Complex*16 array, dimension (lda, n).
 *                      On entry, the LU part of the factorization of the n-by-n
 *                      matrix A computed by CGETC2: A = P * L * U * Q
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in,out] rhs   Complex*16 array, dimension n.
 *                      On entry, the right hand side vector b.
 *                      On exit, the solution vector X.
 * @param[in]     ipiv  The pivot indices; for 0 <= i < n, row i of the
 *                      matrix has been interchanged with row ipiv[i].
 *                      Array of dimension n, 0-based.
 * @param[in]     jpiv  The pivot indices; for 0 <= j < n, column j of the
 *                      matrix has been interchanged with column jpiv[j].
 *                      Array of dimension n, 0-based.
 * @param[out]    scale On exit, SCALE contains the scale factor. SCALE is chosen
 *                      0 <= SCALE <= 1 to prevent overflow in the solution.
 */
void cgesc2(
    const int n,
    const c64* restrict A,
    const int lda,
    c64* restrict rhs,
    const int* restrict ipiv,
    const int* restrict jpiv,
    f32* scale)
{
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    f32 eps, smlnum, bignum;
    c64 temp;
    int i, j, imax;

    if (n == 0) {
        *scale = ONE;
        return;
    }

    /* Set constant to control overflow */

    eps = slamch("P");
    smlnum = slamch("S") / eps;
    bignum = ONE / smlnum;
    (void)bignum;

    /* Apply permutations IPIV to RHS */

    for (i = 0; i < n - 1; i++) {
        if (ipiv[i] != i) {
            temp = rhs[i];
            rhs[i] = rhs[ipiv[i]];
            rhs[ipiv[i]] = temp;
        }
    }

    /* Solve for L part */

    for (i = 0; i < n - 1; i++) {
        for (j = i + 1; j < n; j++) {
            rhs[j] = rhs[j] - A[j + i * lda] * rhs[i];
        }
    }

    /* Solve for U part */

    *scale = ONE;

    /* Check for scaling */

    imax = cblas_icamax(n, rhs, 1);
    if (TWO * smlnum * cabsf(rhs[imax]) > cabsf(A[(n - 1) + (n - 1) * lda])) {
        temp = CMPLXF(ONE / TWO, 0.0f) / cabsf(rhs[imax]);
        cblas_cscal(n, &temp, rhs, 1);
        *scale = (*scale) * crealf(temp);
    }
    for (i = n - 1; i >= 0; i--) {
        temp = CMPLXF(ONE, 0.0f) / A[i + i * lda];
        rhs[i] = rhs[i] * temp;
        for (j = i + 1; j < n; j++) {
            rhs[i] = rhs[i] - rhs[j] * (A[i + j * lda] * temp);
        }
    }

    /* Apply permutations JPIV to the solution (RHS) */

    for (i = n - 2; i >= 0; i--) {
        if (jpiv[i] != i) {
            temp = rhs[i];
            rhs[i] = rhs[jpiv[i]];
            rhs[jpiv[i]] = temp;
        }
    }
}
