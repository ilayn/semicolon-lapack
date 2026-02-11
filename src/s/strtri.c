/**
 * @file strtri.c
 * @brief Computes the inverse of a triangular matrix (blocked algorithm).
 */

#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_single.h"

/**
 * STRTRI computes the inverse of a real upper or lower triangular
 * matrix A.
 *
 * This is the Level 3 BLAS version of the algorithm.
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
 *                           - > 0: if info = i, A(i-1,i-1) is exactly zero. The triangular
 *                           matrix is singular and its inverse cannot be computed.
 */
void strtri(
    const char* uplo,
    const char* diag,
    const int n,
    float * const restrict A,
    const int lda,
    int *info)
{
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    const float NEG_ONE = -1.0f;

    int upper, nounit;
    int j, jb, nb, nn;

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
        xerbla("STRTRI", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) {
        return;
    }

    // Check for singularity if non-unit
    if (nounit) {
        for (*info = 0; *info < n; (*info)++) {
            if (A[*info + *info * lda] == ZERO) {
                (*info)++;  // Convert to 1-based for error reporting
                return;
            }
        }
        *info = 0;
    }

    // Determine the block size for this environment
    nb = lapack_get_nb("TRTRI");

    if (nb <= 1 || nb >= n) {
        // Use unblocked code
        strti2(uplo, diag, n, A, lda, info);
    } else {
        // Use blocked code
        if (upper) {
            // Compute inverse of upper triangular matrix
            for (j = 0; j < n; j += nb) {
                jb = (nb < n - j) ? nb : n - j;

                // Compute rows 0:j-1 of current block column
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                            nounit ? CblasNonUnit : CblasUnit,
                            j, jb, ONE, A, lda, &A[j * lda], lda);
                cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                            nounit ? CblasNonUnit : CblasUnit,
                            j, jb, NEG_ONE, &A[j + j * lda], lda, &A[j * lda], lda);

                // Compute inverse of current diagonal block
                strti2("U", diag, jb, &A[j + j * lda], lda, info);
            }
        } else {
            // Compute inverse of lower triangular matrix
            nn = ((n - 1) / nb) * nb;
            for (j = nn; j >= 0; j -= nb) {
                jb = (nb < n - j) ? nb : n - j;

                if (j + jb < n) {
                    // Compute rows j+jb:n-1 of current block column
                    cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                                nounit ? CblasNonUnit : CblasUnit,
                                n - j - jb, jb, ONE, &A[(j + jb) + (j + jb) * lda], lda,
                                &A[(j + jb) + j * lda], lda);
                    cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                                nounit ? CblasNonUnit : CblasUnit,
                                n - j - jb, jb, NEG_ONE, &A[j + j * lda], lda,
                                &A[(j + jb) + j * lda], lda);
                }

                // Compute inverse of current diagonal block
                strti2("L", diag, jb, &A[j + j * lda], lda, info);
            }
        }
    }
}
