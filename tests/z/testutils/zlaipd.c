/**
 * @file zlaipd.c
 * @brief ZLAIPD sets the imaginary part of the diagonal elements of a complex
 *        matrix A to a large value.
 */

#include "verify.h"

/**
 * ZLAIPD sets the imaginary part of the diagonal elements of a complex
 * matrix A to a large value.  This is used to test LAPACK routines for
 * complex Hermitian matrices, which are not supposed to access or use
 * the imaginary parts of the diagonals.
 *
 * @param[in]     n     The number of diagonal elements of A.
 * @param[in,out] A     On entry, the complex (Hermitian) matrix A.
 *                      On exit, the imaginary parts of the diagonal elements
 *                      are set to BIGNUM = EPS / SAFMIN.
 * @param[in]     inda  The increment between A[0] and the next diagonal element
 *                      of A. Typical values are
 *                      = lda+1: square matrices with leading dimension lda
 *                      = 2: packed upper triangular matrix, starting at A[0]
 *                      = n: packed lower triangular matrix, starting at A[0]
 * @param[in]     vinda The change in the diagonal increment between columns of A.
 *                      Typical values are
 *                      = 0: no change, the row and column increments in A are fixed
 *                      = 1: packed upper triangular matrix
 *                      = -1: packed lower triangular matrix
 */
void zlaipd(const INT n, c128* A, const INT inda, const INT vinda)
{
    f64 bignum = dlamch("E") / dlamch("S");
    INT ia = 0;
    INT ixa = inda;
    for (INT i = 0; i < n; i++) {
        A[ia] = CMPLX(creal(A[ia]), bignum);
        ia += ixa;
        ixa += vinda;
    }
}
