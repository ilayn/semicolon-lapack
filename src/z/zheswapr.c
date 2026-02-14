/**
 * @file zheswapr.c
 * @brief ZHESWAPR applies an elementary permutation on the rows and columns of a Hermitian matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHESWAPR applies an elementary permutation on the rows and the columns of
 * a hermitian matrix.
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
 *          On entry, the NB diagonal matrix D and the multipliers
 *          used to obtain the factor U or L as computed by CSYTRF.
 *
 *          On exit, if INFO = 0, the (symmetric) inverse of the original
 *          matrix.  If UPLO = 'U', the upper triangular part of the
 *          inverse is formed and the part of A below the diagonal is not
 *          referenced; if UPLO = 'L' the lower triangular part of the
 *          inverse is formed and the part of A above the diagonal is
 *          not referenced.
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
void zheswapr(
    const char* uplo,
    const int n,
    c128* restrict A,
    const int lda,
    const int i1,
    const int i2)
{
    int upper;
    int i;
    c128 tmp;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (upper) {

        /* first swap
         *  - swap column i1 and i2 from 0 to i1-1 */
        if (i1 > 0) {
            cblas_zswap(i1, &A[0 + i1 * lda], 1, &A[0 + i2 * lda], 1);
        }

        /* second swap:
         *  - swap A(i1,i1) and A(i2,i2)
         *  - swap row i1 from i1+1 to i2-1 with col i2 from i1+1 to i2-1
         *  - swap A(i2,i1) and A(i1,i2) */
        tmp = A[i1 + i1 * lda];
        A[i1 + i1 * lda] = A[i2 + i2 * lda];
        A[i2 + i2 * lda] = tmp;

        for (i = 1; i <= i2 - i1 - 1; i++) {
            tmp = A[i1 + (i1 + i) * lda];
            A[i1 + (i1 + i) * lda] = conj(A[i1 + i + i2 * lda]);
            A[i1 + i + i2 * lda] = conj(tmp);
        }

        A[i1 + i2 * lda] = conj(A[i1 + i2 * lda]);

        /* third swap
         *  - swap row i1 and i2 from i2+1 to n-1 */
        for (i = i2 + 1; i < n; i++) {
            tmp = A[i1 + i * lda];
            A[i1 + i * lda] = A[i2 + i * lda];
            A[i2 + i * lda] = tmp;
        }

    } else {

        /* first swap
         *  - swap row i1 and i2 from 0 to i1-1 */
        if (i1 > 0) {
            cblas_zswap(i1, &A[i1 + 0 * lda], lda, &A[i2 + 0 * lda], lda);
        }

        /* second swap:
         *  - swap A(i1,i1) and A(i2,i2)
         *  - swap col i1 from i1+1 to i2-1 with row i2 from i1+1 to i2-1
         *  - swap A(i2,i1) and A(i1,i2) */
        tmp = A[i1 + i1 * lda];
        A[i1 + i1 * lda] = A[i2 + i2 * lda];
        A[i2 + i2 * lda] = tmp;

        for (i = 1; i <= i2 - i1 - 1; i++) {
            tmp = A[i1 + i + i1 * lda];
            A[i1 + i + i1 * lda] = conj(A[i2 + (i1 + i) * lda]);
            A[i2 + (i1 + i) * lda] = conj(tmp);
        }

        A[i2 + i1 * lda] = conj(A[i2 + i1 * lda]);

        /* third swap
         *  - swap col i1 and i2 from i2+1 to n-1 */
        for (i = i2 + 1; i < n; i++) {
            tmp = A[i + i1 * lda];
            A[i + i1 * lda] = A[i + i2 * lda];
            A[i + i2 * lda] = tmp;
        }

    }
}
