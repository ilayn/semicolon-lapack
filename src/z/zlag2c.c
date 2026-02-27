/**
 * @file zlag2c.c
 * @brief ZLAG2C converts a complex f64 precision matrix to a complex single precision matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <float.h>

/**
 * ZLAG2C converts a COMPLEX*16 matrix, A, to a COMPLEX matrix, SA.
 *
 * RMAX is the overflow for the SINGLE PRECISION arithmetic.
 * ZLAG2C checks that all the entries of A are between -RMAX and
 * RMAX. If not the conversion is aborted and a flag is raised.
 *
 * This is an auxiliary routine so there is no argument checking.
 *
 * @param[in]     m     The number of lines of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     A     Complex*16 array, dimension (lda, n).
 *                      On entry, the M-by-N coefficient matrix A.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    SA    Complex (single precision) array, dimension (ldsa, n).
 *                      On exit, if info=0, the M-by-N coefficient matrix SA;
 *                      if info>0, the content of SA is unspecified.
 * @param[in]     ldsa  The leading dimension of the array SA. ldsa >= max(1, m).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit.
 *                           - = 1: an entry of the matrix A is greater
 *                           than the SINGLE PRECISION overflow threshold,
 *                           in this case, the content of SA is unspecified.
 */
void zlag2c(
    const INT m,
    const INT n,
    const c128* restrict A,
    const INT lda,
    c64* restrict SA,
    const INT ldsa,
    INT* info)
{
    const f64 rmax = (f64)FLT_MAX;

    *info = 0;

    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < m; i++) {
            f64 re = creal(A[i + j * lda]);
            f64 im = cimag(A[i + j * lda]);
            if ((re < -rmax) || (re > rmax) ||
                (im < -rmax) || (im > rmax)) {
                *info = 1;
                return;
            }
            SA[i + j * ldsa] = CMPLXF((float)re, (float)im);
        }
    }
}
