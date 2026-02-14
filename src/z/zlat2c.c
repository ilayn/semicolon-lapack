/**
 * @file zlat2c.c
 * @brief ZLAT2C converts a double complex triangular matrix to a complex triangular matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <float.h>

/**
 * ZLAT2C converts a COMPLEX*16 triangular matrix, A, to a COMPLEX
 * triangular matrix, SA.
 *
 * RMAX is the overflow for the SINGLE PRECISION arithmetic.
 * ZLAT2C checks that all the entries of A are between -RMAX and
 * RMAX. If not the conversion is aborted and a flag is raised.
 *
 * This is an auxiliary routine so there is no argument checking.
 *
 * @param[in]     uplo    Specifies whether the matrix A is upper or lower
 *                        triangular.
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A.
 *                        n >= 0.
 * @param[in]     A       Complex*16 array, dimension (lda, n).
 *                        On entry, the N-by-N triangular coefficient matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    SA      Complex (single precision) array, dimension (ldsa, n).
 *                        Only the UPLO part of SA is referenced. On exit, if
 *                        info=0, the N-by-N coefficient matrix SA; if info>0,
 *                        the content of the UPLO part of SA is unspecified.
 * @param[in]     ldsa    The leading dimension of the array SA. ldsa >= max(1, n).
 * @param[out]    info
 *                             Exit status:
 *                             - = 0: successful exit.
 *                             - = 1: an entry of the matrix A is greater
 *                             than the SINGLE PRECISION overflow threshold,
 *                             in this case, the content of the UPLO part of
 *                             SA in exit is unspecified.
 */
void zlat2c(
    const char* uplo,
    const int n,
    const double complex* const restrict A,
    const int lda,
    float complex* const restrict SA,
    const int ldsa,
    int* info)
{
    int i, j;
    const double rmax = (double)FLT_MAX;
    int upper;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                double re = creal(A[i + j * lda]);
                double im = cimag(A[i + j * lda]);
                if ((re < -rmax) || (re > rmax) ||
                    (im < -rmax) || (im > rmax)) {
                    *info = 1;
                    return;
                }
                SA[i + j * ldsa] = CMPLXF((float)re, (float)im);
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            for (i = j; i < n; i++) {
                double re = creal(A[i + j * lda]);
                double im = cimag(A[i + j * lda]);
                if ((re < -rmax) || (re > rmax) ||
                    (im < -rmax) || (im > rmax)) {
                    *info = 1;
                    return;
                }
                SA[i + j * ldsa] = CMPLXF((float)re, (float)im);
            }
        }
    }
}
