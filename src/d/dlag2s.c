/**
 * @file dlag2s.c
 * @brief Convert f64 precision matrix to single precision.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DLAG2S converts a DOUBLE PRECISION matrix, A, to a SINGLE
 * PRECISION matrix, SA.
 *
 * RMAX is the overflow for the SINGLE PRECISION arithmetic.
 * DLAG2S checks that all the entries of A are between -RMAX and
 * RMAX. If not the conversion is aborted and a flag is raised.
 *
 * This is an auxiliary routine so there is no argument checking.
 *
 * @param[in]     m     The number of lines of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     A     Double precision array, dimension (lda, n).
 *                      On entry, the M-by-N coefficient matrix A.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    SA    Real (single precision) array, dimension (ldsa, n).
 *                      On exit, if info=0, the M-by-N coefficient matrix SA;
 *                      if info>0, the content of SA is unspecified.
 * @param[in]     ldsa  The leading dimension of the array SA. ldsa >= max(1, m).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit.
 *                           - > 0: if info = 1, an entry of the matrix A is greater
 *                           than the SINGLE PRECISION overflow threshold,
 *                           in this case, the content of SA is unspecified.
 */
void dlag2s(
    const int m,
    const int n,
    const f64* restrict A,
    const int lda,
    float * restrict SA,
    const int ldsa,
    int* info)
{
    // Maximum single precision value
    const f64 rmax = (f64)FLT_MAX;

    *info = 0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            f64 val = A[i + j * lda];
            if (fabs(val) > rmax) {
                *info = 1;
                return;
            }
            SA[i + j * ldsa] = (float)val;
        }
    }
}
