/**
 * @file clag2z.c
 * @brief CLAG2Z converts a complex single precision matrix to a complex double precision matrix.
 */

#include "semicolon_lapack_complex_single.h"

/**
 * CLAG2Z converts a COMPLEX matrix, SA, to a COMPLEX*16 matrix, A.
 *
 * Note that while it is possible to overflow while converting
 * from double to single, it is not possible to overflow when
 * converting from single to double.
 *
 * This is an auxiliary routine so there is no argument checking.
 *
 * @param[in]     m     The number of lines of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     SA    Complex (single precision) array, dimension (ldsa, n).
 *                      On entry, the M-by-N coefficient matrix SA.
 * @param[in]     ldsa  The leading dimension of the array SA. ldsa >= max(1, m).
 * @param[out]    A     Complex*16 array, dimension (lda, n).
 *                      On exit, the M-by-N coefficient matrix A.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit (always succeeds).
 */
void clag2z(
    const INT m,
    const INT n,
    const c64* restrict SA,
    const INT ldsa,
    c128* restrict A,
    const INT lda,
    INT* info)
{
    *info = 0;

    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < m; i++) {
            A[i + j * lda] = (c128)SA[i + j * ldsa];
        }
    }
}
