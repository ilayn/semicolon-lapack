/**
 * @file zpotrf.c
 * @brief ZPOTRF computes the Cholesky factorization of a Hermitian positive
 *        definite matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZPOTRF computes the Cholesky factorization of a complex Hermitian
 * positive definite matrix A.
 *
 * The factorization has the form
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L  * L**H, if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * This is the block version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part of
 *                      the Hermitian matrix A is stored.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
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
void zpotrf(
    const char* uplo,
    const int n,
    c128* restrict A,
    const int lda,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

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
        xerbla("ZPOTRF", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // Determine the block size (NB=64 from ilaenv for POTRF)
    int nb = lapack_get_nb("POTRF");

    if (nb <= 1 || nb >= n) {
        // Use unblocked code
        zpotrf2(uplo, n, A, lda, info);
    } else {
        // Use blocked code
        if (upper) {
            // Compute the Cholesky factorization A = U**H * U.
            for (int j = 0; j < n; j += nb) {
                // Update and factorize the current diagonal block
                int jb = nb < (n - j) ? nb : (n - j);

                // ZHERK('Upper', 'Conjugate transpose', JB, J-1, -ONE, A(1,J), LDA, ONE, A(J,J), LDA)
                if (j > 0) {
                    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                                jb, j, NEG_ONE,
                                &A[j * lda], lda,
                                ONE, &A[j + j * lda], lda);
                }

                // Factor the diagonal block
                zpotrf2("U", jb, &A[j + j * lda], lda, info);
                if (*info != 0) {
                    *info = *info + j;
                    return;
                }

                if (j + jb < n) {
                    // Compute the current block row
                    // ZGEMM('Conjugate transpose', 'No transpose', JB, N-J-JB+1, J-1, -CONE, A(1,J), LDA, A(1,J+JB), LDA, CONE, A(J,J+JB), LDA)
                    if (j > 0) {
                        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                    jb, n - j - jb, j, &NEG_CONE,
                                    &A[j * lda], lda,
                                    &A[(j + jb) * lda], lda,
                                    &CONE, &A[j + (j + jb) * lda], lda);
                    }
                    // ZTRSM('Left', 'Upper', 'Conjugate transpose', 'Non-unit', JB, N-J-JB+1, CONE, A(J,J), LDA, A(J,J+JB), LDA)
                    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                                CblasNonUnit, jb, n - j - jb, &CONE,
                                &A[j + j * lda], lda,
                                &A[j + (j + jb) * lda], lda);
                }
            }
        } else {
            // Compute the Cholesky factorization A = L * L**H.
            for (int j = 0; j < n; j += nb) {
                // Update and factorize the current diagonal block
                int jb = nb < (n - j) ? nb : (n - j);

                // ZHERK('Lower', 'No transpose', JB, J-1, -ONE, A(J,1), LDA, ONE, A(J,J), LDA)
                if (j > 0) {
                    cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
                                jb, j, NEG_ONE,
                                &A[j], lda,
                                ONE, &A[j + j * lda], lda);
                }

                // Factor the diagonal block
                zpotrf2("L", jb, &A[j + j * lda], lda, info);
                if (*info != 0) {
                    *info = *info + j;
                    return;
                }

                if (j + jb < n) {
                    // Compute the current block column
                    // ZGEMM('No transpose', 'Conjugate transpose', N-J-JB+1, JB, J-1, -CONE, A(J+JB,1), LDA, A(J,1), LDA, CONE, A(J+JB,J), LDA)
                    if (j > 0) {
                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                    n - j - jb, jb, j, &NEG_CONE,
                                    &A[j + jb], lda,
                                    &A[j], lda,
                                    &CONE, &A[(j + jb) + j * lda], lda);
                    }
                    // ZTRSM('Right', 'Lower', 'Conjugate transpose', 'Non-unit', N-J-JB+1, JB, CONE, A(J,J), LDA, A(J+JB,J), LDA)
                    cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                                CblasNonUnit, n - j - jb, jb, &CONE,
                                &A[j + j * lda], lda,
                                &A[(j + jb) + j * lda], lda);
                }
            }
        }
    }
}
