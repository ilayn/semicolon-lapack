/**
 * @file strti2.c
 * @brief Computes the inverse of a triangular matrix (unblocked algorithm).
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STRTI2 computes the inverse of a real upper or lower triangular
 * matrix.
 *
 * This is the Level 2 BLAS version of the algorithm.
 *
 * @param[in]     uplo  'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     diag  'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     n     The order of the matrix A (n >= 0).
 * @param[in,out] A     On entry, the triangular matrix A.
 *                      On exit, the (triangular) inverse of the original matrix.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,n)).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -k, the k-th argument had an illegal value
 */
void strti2(
    const char* uplo,
    const char* diag,
    const int n,
    f32 * const restrict A,
    const int lda,
    int *info)
{
    const f32 ONE = 1.0f;

    int upper, nounit;
    int j;
    f32 ajj;

    // Test the input parameters
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

    if (!upper && uplo[0] != 'L' && uplo[0] != 'l') {
        *info = -1;
    } else if (!nounit && diag[0] != 'U' && diag[0] != 'u') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    }

    if (*info != 0) {
        xerbla("STRTI2", -(*info));
        return;
    }

    if (upper) {
        // Compute inverse of upper triangular matrix
        for (j = 0; j < n; j++) {
            if (nounit) {
                A[j + j * lda] = ONE / A[j + j * lda];
                ajj = -A[j + j * lda];
            } else {
                ajj = -ONE;
            }

            // Compute elements 0:j-1 of j-th column
            cblas_strmv(CblasColMajor, CblasUpper, CblasNoTrans,
                        nounit ? CblasNonUnit : CblasUnit,
                        j, A, lda, &A[j * lda], 1);
            cblas_sscal(j, ajj, &A[j * lda], 1);
        }
    } else {
        // Compute inverse of lower triangular matrix
        for (j = n - 1; j >= 0; j--) {
            if (nounit) {
                A[j + j * lda] = ONE / A[j + j * lda];
                ajj = -A[j + j * lda];
            } else {
                ajj = -ONE;
            }

            if (j < n - 1) {
                // Compute elements j+1:n-1 of j-th column
                cblas_strmv(CblasColMajor, CblasLower, CblasNoTrans,
                            nounit ? CblasNonUnit : CblasUnit,
                            n - j - 1, &A[(j + 1) + (j + 1) * lda], lda,
                            &A[(j + 1) + j * lda], 1);
                cblas_sscal(n - j - 1, ajj, &A[(j + 1) + j * lda], 1);
            }
        }
    }
}
