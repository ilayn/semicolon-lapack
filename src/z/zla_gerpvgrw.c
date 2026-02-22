/**
 * @file zla_gerpvgrw.c
 * @brief ZLA_GERPVGRW computes the reciprocal pivot growth factor
 *        norm(A)/norm(U).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLA_GERPVGRW computes the reciprocal pivot growth factor
 * norm(A)/norm(U). The "max absolute element" norm is used. If this is
 * much less than 1, the stability of the LU factorization of the
 * (equilibrated) matrix A could be poor. This also means that the
 * solution X, estimated condition numbers, and error bounds could be
 * unreliable.
 *
 * @param[in] n     The number of linear equations, i.e., the order of the
 *                  matrix A. n >= 0.
 * @param[in] ncols The number of columns of the matrix A. ncols >= 0.
 * @param[in] A     Complex array, dimension (lda, n).
 *                  On entry, the N-by-N matrix A.
 * @param[in] lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in] AF    Complex array, dimension (ldaf, n).
 *                  The factors L and U from the factorization
 *                  A = P*L*U as computed by ZGETRF.
 * @param[in] ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @return    The reciprocal pivot growth factor.
 */
f64 zla_gerpvgrw(
    const INT n,
    const INT ncols,
    const c128* restrict A,
    const INT lda,
    const c128* restrict AF,
    const INT ldaf)
{
    INT i, j;
    f64 amax, umax, rpvgrw;

    rpvgrw = 1.0;

    for (j = 0; j < ncols; j++) {
        amax = 0.0;
        umax = 0.0;
        for (i = 0; i < n; i++) {
            amax = fmax(cabs1(A[i + j * lda]), amax);
        }
        for (i = 0; i <= j; i++) {
            umax = fmax(cabs1(AF[i + j * ldaf]), umax);
        }
        if (umax != 0.0) {
            rpvgrw = fmin(amax / umax, rpvgrw);
        }
    }
    return rpvgrw;
}
