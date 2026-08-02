/**
 * @file zheswapr.c
 * @brief ZHESWAPR applies an elementary permutation on the rows and columns of a Hermitian matrix.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZHESWAPR applies an elementary permutation on the rows and the columns of
 * a Hermitian matrix.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangular part of A is stored
 *                      - `'L'`: Lower triangular part of A is stored
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Double complex array of dimension `(lda,n)`.
 *                      On entry, the Hermitian matrix A. On exit, rows
 *                      `i1` and `i2` and columns `i1` and `i2` are
 *                      interchanged. The unused triangle is not referenced.
 * @param[in]     lda   The leading dimension of A. `lda>=max(1,n)`.
 * @param[in]     i1    The 0-based index of the first row and column.
 * @param[in]     i2    The 0-based index of the second row and column.
 */
void zheswapr(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    const INT i1,
    const INT i2)
{
    INT upper;
    INT i;
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
