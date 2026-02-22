/**
 * @file zgesc2.c
 * @brief ZGESC2 solves a system of linear equations using the LU factorization
 *        with complete pivoting computed by ZGETC2.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZGESC2 solves a system of linear equations
 *
 *           A * X = scale * RHS
 *
 * with a general N-by-N matrix A using the LU factorization with
 * complete pivoting computed by ZGETC2.
 *
 * @param[in]     n     The number of columns of the matrix A.
 * @param[in]     A     Complex*16 array, dimension (lda, n).
 *                      On entry, the LU part of the factorization of the n-by-n
 *                      matrix A computed by ZGETC2: A = P * L * U * Q
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
void zgesc2(
    const INT n,
    const c128* restrict A,
    const INT lda,
    c128* restrict rhs,
    const INT* restrict ipiv,
    const INT* restrict jpiv,
    f64* scale)
{
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 eps, smlnum, bignum;
    c128 temp;
    INT i, j, imax;

    if (n == 0) {
        *scale = ONE;
        return;
    }

    /* Set constant to control overflow */

    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
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

    imax = cblas_izamax(n, rhs, 1);
    if (TWO * smlnum * cabs(rhs[imax]) > cabs(A[(n - 1) + (n - 1) * lda])) {
        temp = CMPLX(ONE / TWO, 0.0) / cabs(rhs[imax]);
        cblas_zscal(n, &temp, rhs, 1);
        *scale = (*scale) * creal(temp);
    }
    for (i = n - 1; i >= 0; i--) {
        temp = CMPLX(ONE, 0.0) / A[i + i * lda];
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
