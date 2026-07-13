/**
 * @file dla_gerpvgrw.c
 * @brief DLA_GERPVGRW computes the reciprocal pivot growth factor
 *        norm(A)/norm(U).
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLA_GERPVGRW computes the reciprocal pivot growth factor
 * norm(A)/norm(U). The "max absolute element" norm is used. If this is
 * much less than 1, the stability of the LU factorization of the
 * (equilibrated) matrix A could be poor. This also means that the
 * solution X, estimated condition numbers, and error bounds could be
 * unreliable.
 *
 * @param[in] n     The number of linear equations, i.e., the order of the
 *                  matrix A. n >= 0.
 * @param[in] ncols The number of columns of the matrix A. ncols >= 0.
 * @param[in] A     Double precision array, dimension (lda, n).
 *                  On entry, the N-by-N matrix A.
 * @param[in] lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in] AF    Double precision array, dimension (ldaf, n).
 *                  The factors L and U from the factorization
 *                  A = P*L*U as computed by DGETRF.
 * @param[in] ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @return    The reciprocal pivot growth factor.
 */
f64 dla_gerpvgrw(
    const INT n,
    const INT ncols,
    const f64* restrict A,
    const INT lda,
    const f64* restrict AF,
    const INT ldaf)
{
    INT i, j;
    f64 amax, umax, rpvgrw;

    rpvgrw = 1.0;

    for (j = 0; j < ncols; j++) {
        amax = 0.0;
        umax = 0.0;
        for (i = 0; i < n; i++) {
            amax = fmax(fabs(A[i + j * lda]), amax);
        }
        for (i = 0; i <= j; i++) {
            umax = fmax(fabs(AF[i + j * ldaf]), umax);
        }
        if (umax != 0.0) {
            rpvgrw = fmin(amax / umax, rpvgrw);
        }
    }
    return rpvgrw;
}
