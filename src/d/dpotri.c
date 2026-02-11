/**
 * @file dpotri.c
 * @brief DPOTRI computes the inverse of a symmetric positive definite matrix.
 */

#include "semicolon_lapack_double.h"

/**
 * DPOTRI computes the inverse of a real symmetric positive definite
 * matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
 * computed by DPOTRF.
 *
 * @param[in]     uplo  Specifies whether the factor stored in A is upper or
 *                      lower triangular.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     On entry, the triangular factor U or L from the
 *                      Cholesky factorization A = U**T*U or A = L*L**T,
 *                      as computed by dpotrf.
 *                      On exit, the upper or lower triangle of the (symmetric)
 *                      inverse of A, overwriting the input factor U or L.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the (k,k) element of the factor U or L
 *                           is zero, and the inverse could not be computed.
 */
void dpotri(
    const char* uplo,
    const int n,
    double* const restrict A,
    const int lda,
    int* info)
{
    // Test the input parameters
    *info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DPOTRI", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) return;

    // Invert the triangular Cholesky factor U or L.
    dtrtri(uplo, "N", n, A, lda, info);
    if (*info > 0) return;

    // Form inv(U) * inv(U)**T or inv(L)**T * inv(L).
    dlauum(uplo, n, A, lda, info);
}
