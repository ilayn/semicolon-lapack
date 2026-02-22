/**
 * @file slauu2.c
 * @brief SLAUU2 computes the product U * U**T or L**T * L (unblocked algorithm).
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLAUU2 computes the product U * U**T or L**T * L, where the triangular
 * factor U or L is stored in the upper or lower triangular part of
 * the array A.
 *
 * If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
 * overwriting the factor U in A.
 * If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
 * overwriting the factor L in A.
 *
 * This is the unblocked form of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo  Specifies whether the triangular factor stored in the
 *                      array A is upper or lower triangular:
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the triangular factor U or L. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the triangular factor U or L.
 *                      On exit, if UPLO = 'U', the upper triangle of A is
 *                      overwritten with the upper triangle of the product U * U**T;
 *                      if UPLO = 'L', the lower triangle of A is overwritten with
 *                      the lower triangle of the product L**T * L.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void slauu2(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    INT* info)
{
    const f32 ONE = 1.0f;

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
        xerbla("SLAUU2", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    if (upper) {
        // Compute the product U * U**T.
        for (INT i = 0; i < n; i++) {
            f32 aii = A[i + i * lda];
            if (i < n - 1) {
                // A(i,i) = dot(row i from col i to n-1)
                // Fortran: DDOT(N-I+1, A(I,I), LDA, A(I,I), LDA)
                // In 0-based: length = n - i, stride = lda
                A[i + i * lda] = cblas_sdot(n - i, &A[i + i * lda], lda,
                                            &A[i + i * lda], lda);
                // Fortran: DGEMV('N', I-1, N-I, ONE, A(1,I+1), LDA, A(I,I+1), LDA, AII, A(1,I), 1)
                // 0-based: m = i, n_cols = n-i-1
                cblas_sgemv(CblasColMajor, CblasNoTrans,
                            i, n - i - 1, ONE,
                            &A[(i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda,
                            aii, &A[i * lda], 1);
            } else {
                // Fortran: DSCAL(I, AII, A(1,I), 1)
                // 0-based: length = i+1 (scale the entire column up to and including diagonal)
                cblas_sscal(i + 1, aii, &A[i * lda], 1);
            }
        }
    } else {
        // Compute the product L**T * L.
        for (INT i = 0; i < n; i++) {
            f32 aii = A[i + i * lda];
            if (i < n - 1) {
                // A(i,i) = dot(column i from row i to n-1)
                // Fortran: DDOT(N-I+1, A(I,I), 1, A(I,I), 1)
                // 0-based: length = n - i, stride = 1
                A[i + i * lda] = cblas_sdot(n - i, &A[i + i * lda], 1,
                                            &A[i + i * lda], 1);
                // Fortran: DGEMV('T', N-I, I-1, ONE, A(I+1,1), LDA, A(I+1,I), 1, AII, A(I,1), LDA)
                // 0-based: m = n-i-1, n_cols = i
                cblas_sgemv(CblasColMajor, CblasTrans,
                            n - i - 1, i, ONE,
                            &A[(i + 1)], lda,
                            &A[(i + 1) + i * lda], 1,
                            aii, &A[i], lda);
            } else {
                // Fortran: DSCAL(I, AII, A(I,1), LDA)
                // 0-based: length = i+1 (scale the entire row up to and including diagonal)
                cblas_sscal(i + 1, aii, &A[i], lda);
            }
        }
    }
}
