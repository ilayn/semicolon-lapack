/**
 * @file dpotrf2.c
 * @brief DPOTRF2 computes the Cholesky factorization of a symmetric positive
 *        definite matrix using the recursive algorithm.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPOTRF2 computes the Cholesky factorization of a real symmetric
 * positive definite matrix A using the recursive algorithm.
 *
 * The factorization has the form
 *    A = U**T * U,  if UPLO = 'U', or
 *    A = L  * L**T, if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the recursive version of the algorithm. It divides
 * the matrix into four submatrices:
 *
 *        [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
 *    A = [ -----|----- ]  with n1 = n/2
 *        [  A21 | A22  ]       n2 = n-n1
 *
 * The subroutine calls itself to factor A11. Update and scale A21
 * or A12, update A22 then calls itself to factor A22.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part of
 *                      the symmetric matrix A is stored.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the symmetric matrix A.
 *                      On exit, if info = 0, the factor U or L from the
 *                      Cholesky factorization A = U**T*U or A = L*L**T.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           is not positive, and the factorization could not be
 *                           completed.
 */
void dpotrf2(
    const char* uplo,
    const int n,
    f64* restrict A,
    const int lda,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;
    const f64 ZERO = 0.0;

    // Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DPOTRF2", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // N=1 case
    if (n == 1) {
        // Test for non-positive-definiteness
        if (A[0] <= ZERO || disnan(A[0])) {
            *info = 1;
            return;
        }
        // Factor
        A[0] = sqrt(A[0]);
    } else {
        // Use recursive code
        int n1 = n / 2;
        int n2 = n - n1;
        int iinfo;

        // Factor A11
        dpotrf2(uplo, n1, A, lda, &iinfo);
        if (iinfo != 0) {
            *info = iinfo;
            return;
        }

        if (upper) {
            // Compute the Cholesky factorization A = U**T * U

            // Update and scale A12
            // Fortran: DTRSM('L', 'U', 'T', 'N', N1, N2, ONE, A(1,1), LDA, A(1,N1+1), LDA)
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                        CblasNonUnit, n1, n2, ONE,
                        A, lda, &A[n1 * lda], lda);

            // Update A22
            // Fortran: DSYRK(UPLO, 'T', N2, N1, -ONE, A(1,N1+1), LDA, ONE, A(N1+1,N1+1), LDA)
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                        n2, n1, NEG_ONE,
                        &A[n1 * lda], lda,
                        ONE, &A[n1 + n1 * lda], lda);

            // Factor A22
            dpotrf2(uplo, n2, &A[n1 + n1 * lda], lda, &iinfo);
            if (iinfo != 0) {
                *info = iinfo + n1;
                return;
            }
        } else {
            // Compute the Cholesky factorization A = L * L**T

            // Update and scale A21
            // Fortran: DTRSM('R', 'L', 'T', 'N', N2, N1, ONE, A(1,1), LDA, A(N1+1,1), LDA)
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                        CblasNonUnit, n2, n1, ONE,
                        A, lda, &A[n1], lda);

            // Update A22
            // Fortran: DSYRK(UPLO, 'N', N2, N1, -ONE, A(N1+1,1), LDA, ONE, A(N1+1,N1+1), LDA)
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        n2, n1, NEG_ONE,
                        &A[n1], lda,
                        ONE, &A[n1 + n1 * lda], lda);

            // Factor A22
            dpotrf2(uplo, n2, &A[n1 + n1 * lda], lda, &iinfo);
            if (iinfo != 0) {
                *info = iinfo + n1;
                return;
            }
        }
    }
}
