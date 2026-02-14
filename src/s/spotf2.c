/**
 * @file spotf2.c
 * @brief SPOTF2 computes the Cholesky factorization of a symmetric positive
 *        definite matrix (unblocked algorithm).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPOTF2 computes the Cholesky factorization of a real symmetric
 * positive definite matrix A.
 *
 * The factorization has the form
 *    A = U**T * U,  if UPLO = 'U', or
 *    A = L  * L**T, if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part of
 *                      the symmetric matrix A is stored.
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the symmetric matrix A. If UPLO = 'U', the
 *                      leading n by n upper triangular part of A contains the
 *                      upper triangular part of the matrix A. If UPLO = 'L',
 *                      the leading n by n lower triangular part of A contains
 *                      the lower triangular part of the matrix A.
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
void spotf2(
    const char* uplo,
    const int n,
    f32* restrict A,
    const int lda,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;
    const f32 ZERO = 0.0f;

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
        xerbla("SPOTF2", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    if (upper) {
        // Compute the Cholesky factorization A = U**T * U.
        for (int j = 0; j < n; j++) {
            // Compute U(j,j) and test for non-positive-definiteness.
            // Fortran: AJJ = A(J,J) - DDOT(J-1, A(1,J), 1, A(1,J), 1)
            // 0-based: length = j (number of elements above diagonal in column j)
            f32 ajj = A[j + j * lda];
            if (j > 0) {
                ajj -= cblas_sdot(j, &A[j * lda], 1, &A[j * lda], 1);
            }
            if (ajj <= ZERO || sisnan(ajj)) {
                A[j + j * lda] = ajj;
                *info = j + 1;  // 1-based for error reporting
                return;
            }
            ajj = sqrtf(ajj);
            A[j + j * lda] = ajj;

            // Compute elements j+1:n-1 of row j.
            if (j < n - 1) {
                // Fortran: DGEMV('T', J-1, N-J, -ONE, A(1,J+1), LDA, A(1,J), 1, ONE, A(J,J+1), LDA)
                // 0-based: m = j, n_cols = n-j-1
                if (j > 0) {
                    cblas_sgemv(CblasColMajor, CblasTrans,
                                j, n - j - 1, NEG_ONE,
                                &A[(j + 1) * lda], lda,
                                &A[j * lda], 1,
                                ONE, &A[j + (j + 1) * lda], lda);
                }
                // Fortran: DSCAL(N-J, ONE/AJJ, A(J,J+1), LDA)
                cblas_sscal(n - j - 1, ONE / ajj, &A[j + (j + 1) * lda], lda);
            }
        }
    } else {
        // Compute the Cholesky factorization A = L * L**T.
        for (int j = 0; j < n; j++) {
            // Compute L(j,j) and test for non-positive-definiteness.
            // Fortran: AJJ = A(J,J) - DDOT(J-1, A(J,1), LDA, A(J,1), LDA)
            // 0-based: length = j (number of elements left of diagonal in row j)
            f32 ajj = A[j + j * lda];
            if (j > 0) {
                ajj -= cblas_sdot(j, &A[j], lda, &A[j], lda);
            }
            if (ajj <= ZERO || sisnan(ajj)) {
                A[j + j * lda] = ajj;
                *info = j + 1;  // 1-based for error reporting
                return;
            }
            ajj = sqrtf(ajj);
            A[j + j * lda] = ajj;

            // Compute elements j+1:n-1 of column j.
            if (j < n - 1) {
                // Fortran: DGEMV('N', N-J, J-1, -ONE, A(J+1,1), LDA, A(J,1), LDA, ONE, A(J+1,J), 1)
                // 0-based: m = n-j-1, n_cols = j
                if (j > 0) {
                    cblas_sgemv(CblasColMajor, CblasNoTrans,
                                n - j - 1, j, NEG_ONE,
                                &A[j + 1], lda,
                                &A[j], lda,
                                ONE, &A[(j + 1) + j * lda], 1);
                }
                // Fortran: DSCAL(N-J, ONE/AJJ, A(J+1,J), 1)
                cblas_sscal(n - j - 1, ONE / ajj, &A[(j + 1) + j * lda], 1);
            }
        }
    }
}
