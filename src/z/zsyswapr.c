/**
 * @file zsyswapr.c
 * @brief ZSYSWAPR applies an elementary permutation on the rows and columns of a symmetric matrix.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSYSWAPR applies an elementary permutation on the rows and the columns of
 * a symmetric matrix.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the N-by-N matrix A. On exit, the permuted matrix
 *          where the rows i1 and i2 and columns i1 and i2 are interchanged.
 *          If UPLO = 'U', the interchanges are applied to the upper
 *          triangular part and the strictly lower triangular part of A is
 *          not referenced; if UPLO = 'L', the interchanges are applied to
 *          the lower triangular part and the part of A above the diagonal
 *          is not referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] i1
 *          Index of the first row to swap (0-based).
 *
 * @param[in] i2
 *          Index of the second row to swap (0-based).
 */
void zsyswapr(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    const INT i1,
    const INT i2)
{
    INT upper;
    c128 tmp;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (upper) {

        if (i1 > 0) {
            cblas_zswap(i1, &A[0 + i1 * lda], 1, &A[0 + i2 * lda], 1);
        }

        tmp = A[i1 + i1 * lda];
        A[i1 + i1 * lda] = A[i2 + i2 * lda];
        A[i2 + i2 * lda] = tmp;

        if (i2 - i1 - 1 > 0) {
            cblas_zswap(i2 - i1 - 1, &A[i1 + (i1 + 1) * lda], lda, &A[i1 + 1 + i2 * lda], 1);
        }

        if (i2 < n - 1) {
            cblas_zswap(n - i2 - 1, &A[i1 + (i2 + 1) * lda], lda, &A[i2 + (i2 + 1) * lda], lda);
        }

    } else {

        if (i1 > 0) {
            cblas_zswap(i1, &A[i1 + 0 * lda], lda, &A[i2 + 0 * lda], lda);
        }

        tmp = A[i1 + i1 * lda];
        A[i1 + i1 * lda] = A[i2 + i2 * lda];
        A[i2 + i2 * lda] = tmp;

        if (i2 - i1 - 1 > 0) {
            cblas_zswap(i2 - i1 - 1, &A[i1 + 1 + i1 * lda], 1, &A[i2 + (i1 + 1) * lda], lda);
        }

        if (i2 < n - 1) {
            cblas_zswap(n - i2 - 1, &A[i2 + 1 + i1 * lda], 1, &A[i2 + 1 + i2 * lda], 1);
        }
    }
}
