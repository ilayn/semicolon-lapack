/**
 * @file spotrf.c
 * @brief SPOTRF computes the Cholesky factorization of a symmetric positive
 *        definite matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SPOTRF computes the Cholesky factorization of a real symmetric
 * positive definite matrix A.
 *
 * The factorization has the form
 *    A = U**T * U,  if UPLO = 'U', or
 *    A = L  * L**T, if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the block version of the algorithm, calling Level 3 BLAS.
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
void spotrf(
    const char* uplo,
    const int n,
    f32* restrict A,
    const int lda,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;

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
        xerbla("SPOTRF", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // Determine the block size (NB=64 from ilaenv for POTRF)
    int nb = lapack_get_nb("POTRF");

    if (nb <= 1 || nb >= n) {
        // Use unblocked code
        spotrf2(uplo, n, A, lda, info);
    } else {
        // Use blocked code
        if (upper) {
            // Compute the Cholesky factorization A = U**T * U.
            for (int j = 0; j < n; j += nb) {
                // Update and factorize the current diagonal block
                int jb = nb < (n - j) ? nb : (n - j);

                // Fortran: DSYRK('Upper', 'T', JB, J-1, -ONE, A(1,J), LDA, ONE, A(J,J), LDA)
                // 0-based: n_size = jb, k = j
                if (j > 0) {
                    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                                jb, j, NEG_ONE,
                                &A[j * lda], lda,
                                ONE, &A[j + j * lda], lda);
                }

                // Factor the diagonal block
                spotrf2("U", jb, &A[j + j * lda], lda, info);
                if (*info != 0) {
                    *info = *info + j;
                    return;
                }

                if (j + jb < n) {
                    // Compute the current block row
                    // Fortran: DGEMM('T', 'N', JB, N-J-JB+1, J-1, -ONE, A(1,J), LDA, A(1,J+JB), LDA, ONE, A(J,J+JB), LDA)
                    if (j > 0) {
                        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                    jb, n - j - jb, j, NEG_ONE,
                                    &A[j * lda], lda,
                                    &A[(j + jb) * lda], lda,
                                    ONE, &A[j + (j + jb) * lda], lda);
                    }
                    // Fortran: DTRSM('L', 'U', 'T', 'N', JB, N-J-JB+1, ONE, A(J,J), LDA, A(J,J+JB), LDA)
                    cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                                CblasNonUnit, jb, n - j - jb, ONE,
                                &A[j + j * lda], lda,
                                &A[j + (j + jb) * lda], lda);
                }
            }
        } else {
            // Compute the Cholesky factorization A = L * L**T.
            for (int j = 0; j < n; j += nb) {
                // Update and factorize the current diagonal block
                int jb = nb < (n - j) ? nb : (n - j);

                // Fortran: DSYRK('Lower', 'N', JB, J-1, -ONE, A(J,1), LDA, ONE, A(J,J), LDA)
                if (j > 0) {
                    cblas_ssyrk(CblasColMajor, CblasLower, CblasNoTrans,
                                jb, j, NEG_ONE,
                                &A[j], lda,
                                ONE, &A[j + j * lda], lda);
                }

                // Factor the diagonal block
                spotrf2("L", jb, &A[j + j * lda], lda, info);
                if (*info != 0) {
                    *info = *info + j;
                    return;
                }

                if (j + jb < n) {
                    // Compute the current block column
                    // Fortran: DGEMM('N', 'T', N-J-JB+1, JB, J-1, -ONE, A(J+JB,1), LDA, A(J,1), LDA, ONE, A(J+JB,J), LDA)
                    if (j > 0) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    n - j - jb, jb, j, NEG_ONE,
                                    &A[j + jb], lda,
                                    &A[j], lda,
                                    ONE, &A[(j + jb) + j * lda], lda);
                    }
                    // Fortran: DTRSM('R', 'L', 'T', 'N', N-J-JB+1, JB, ONE, A(J,J), LDA, A(J+JB,J), LDA)
                    cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                                CblasNonUnit, n - j - jb, jb, ONE,
                                &A[j + j * lda], lda,
                                &A[(j + jb) + j * lda], lda);
                }
            }
        }
    }
}
