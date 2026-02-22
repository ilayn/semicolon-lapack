/**
 * @file stpttr.c
 * @brief STPTTR copies a triangular matrix from standard packed format (TP) to standard full format (TR).
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"

/**
 * STPTTR copies a triangular matrix A from standard packed format (TP)
 * to standard full format (TR).
 *
 * @param[in] uplo
 *          = 'U':  A is upper triangular.
 *          = 'L':  A is lower triangular.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] AP
 *          Double precision array, dimension ( n*(n+1)/2 ),
 *          On entry, the upper or lower triangular matrix A, packed
 *          columnwise in a linear array. The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *
 * @param[out] A
 *          Double precision array, dimension ( lda, n )
 *          On exit, the triangular matrix A.  If UPLO = 'U', the leading
 *          n-by-n upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading n-by-n lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A.  lda >= max(1,n).
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void stpttr(
    const char* uplo,
    const INT n,
    const f32* restrict AP,
    f32* restrict A,
    const INT lda,
    INT* info)
{
    INT lower;
    INT i, j, k;

    *info = 0;
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("STPTTR", -(*info));
        return;
    }

    if (lower) {
        k = 0;
        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                A[i + j * lda] = AP[k];
                k++;
            }
        }
    } else {
        k = 0;
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                A[i + j * lda] = AP[k];
                k++;
            }
        }
    }
}
