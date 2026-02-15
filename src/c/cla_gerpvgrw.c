/**
 * @file cla_gerpvgrw.c
 * @brief CLA_GERPVGRW computes the reciprocal pivot growth factor
 *        norm(A)/norm(U).
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLA_GERPVGRW computes the reciprocal pivot growth factor
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
 *                  A = P*L*U as computed by CGETRF.
 * @param[in] ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @return    The reciprocal pivot growth factor.
 */
f32 cla_gerpvgrw(
    const int n,
    const int ncols,
    const c64* restrict A,
    const int lda,
    const c64* restrict AF,
    const int ldaf)
{
    int i, j;
    f32 amax, umax, rpvgrw;

    rpvgrw = 1.0f;

    for (j = 0; j < ncols; j++) {
        amax = 0.0f;
        umax = 0.0f;
        for (i = 0; i < n; i++) {
            amax = fmaxf(cabs1f(A[i + j * lda]), amax);
        }
        for (i = 0; i <= j; i++) {
            umax = fmaxf(cabs1f(AF[i + j * ldaf]), umax);
        }
        if (umax != 0.0f) {
            rpvgrw = fminf(amax / umax, rpvgrw);
        }
    }
    return rpvgrw;
}
