/**
 * @file dlat2s.c
 * @brief Convert f64 precision triangular matrix to single precision.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DLAT2S converts a DOUBLE PRECISION triangular matrix, A, to a SINGLE
 * PRECISION triangular matrix, SA.
 *
 * RMAX is the overflow for the SINGLE PRECISION arithmetic.
 * DLAT2S checks that all the entries of A are between -RMAX and
 * RMAX. If not the conversion is aborted and a flag is raised.
 *
 * This is an auxiliary routine so there is no argument checking.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the symmetric matrix A is stored.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A     Double precision array, dimension (lda, n).
 *                      On entry, the N-by-N triangular coefficient matrix A.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    SA    Real (single precision) array, dimension (ldsa, n).
 *                      Only the UPLO part of SA is referenced. On exit, if info=0,
 *                      the N-by-N coefficient matrix SA; if info>0, the content of
 *                      the UPLO part of SA is unspecified.
 * @param[in]     ldsa  The leading dimension of the array SA. ldsa >= max(1, n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit.
 *                           - = 1: an entry of the matrix A is greater than the
 *                           SINGLE PRECISION overflow threshold, in this case,
 *                           the content of the UPLO part of SA is unspecified.
 */
void dlat2s(
    const char* uplo,
    const int n,
    const f64* const restrict A,
    const int lda,
    float* const restrict SA,
    const int ldsa,
    int* info)
{
    const f64 rmax = (f64)FLT_MAX;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');

    *info = 0;

    if (upper) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                f64 val = A[i + j * lda];
                if (val < -rmax || val > rmax) {
                    *info = 1;
                    return;
                }
                SA[i + j * ldsa] = (float)val;
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                f64 val = A[i + j * lda];
                if (val < -rmax || val > rmax) {
                    *info = 1;
                    return;
                }
                SA[i + j * ldsa] = (float)val;
            }
        }
    }
}
