/**
 * @file cpotrf2.c
 * @brief CPOTRF2 computes the Cholesky factorization of a Hermitian positive
 *        definite matrix using the recursive algorithm.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPOTRF2 computes the Cholesky factorization of a Hermitian
 * positive definite matrix A using the recursive algorithm.
 *
 * The factorization has the form
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L  * L**H, if UPLO = 'L',
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
 *                      the Hermitian matrix A is stored.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A.
 *                      On exit, if info = 0, the factor U or L from the
 *                      Cholesky factorization A = U**H*U or A = L*L**H.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           is not positive, and the factorization could not be
 *                           completed.
 */
void cpotrf2(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;
    const f32 ZERO = 0.0f;

    // Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CPOTRF2", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // N=1 case
    if (n == 1) {
        // Test for non-positive-definiteness
        f32 ajj = crealf(A[0]);
        if (ajj <= ZERO || sisnan(ajj)) {
            *info = 1;
            return;
        }
        // Factor
        A[0] = sqrtf(ajj);
    } else {
        // Use recursive code
        INT n1 = n / 2;
        INT n2 = n - n1;
        INT iinfo;

        // Factor A11
        cpotrf2(uplo, n1, A, lda, &iinfo);
        if (iinfo != 0) {
            *info = iinfo;
            return;
        }

        if (upper) {
            // Compute the Cholesky factorization A = U**H * U

            // Update and scale A12
            // Fortran: ZTRSM('L', 'U', 'C', 'N', N1, N2, CONE, A(1,1), LDA, A(1,N1+1), LDA)
            cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                        CblasNonUnit, n1, n2, &CONE,
                        A, lda, &A[n1 * lda], lda);

            // Update A22
            // Fortran: ZHERK(UPLO, 'C', N2, N1, -ONE, A(1,N1+1), LDA, ONE, A(N1+1,N1+1), LDA)
            cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                        n2, n1, NEG_ONE,
                        &A[n1 * lda], lda,
                        ONE, &A[n1 + n1 * lda], lda);

            // Factor A22
            cpotrf2(uplo, n2, &A[n1 + n1 * lda], lda, &iinfo);
            if (iinfo != 0) {
                *info = iinfo + n1;
                return;
            }
        } else {
            // Compute the Cholesky factorization A = L * L**H

            // Update and scale A21
            // Fortran: ZTRSM('R', 'L', 'C', 'N', N2, N1, CONE, A(1,1), LDA, A(N1+1,1), LDA)
            cblas_ctrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                        CblasNonUnit, n2, n1, &CONE,
                        A, lda, &A[n1], lda);

            // Update A22
            // Fortran: ZHERK(UPLO, 'N', N2, N1, -ONE, A(N1+1,1), LDA, ONE, A(N1+1,N1+1), LDA)
            cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                        n2, n1, NEG_ONE,
                        &A[n1], lda,
                        ONE, &A[n1 + n1 * lda], lda);

            // Factor A22
            cpotrf2(uplo, n2, &A[n1 + n1 * lda], lda, &iinfo);
            if (iinfo != 0) {
                *info = iinfo + n1;
                return;
            }
        }
    }
}
