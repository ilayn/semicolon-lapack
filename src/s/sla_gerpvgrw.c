/**
 * @file sla_gerpvgrw.c
 * @brief SLA_GERPVGRW computes the reciprocal pivot growth factor
 *        norm(A)/norm(U).
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLA_GERPVGRW computes the reciprocal pivot growth factor
 * norm(A)/norm(U). The "max absolute element" norm is used. If this is
 * much less than 1, the stability of the LU factorization of the
 * (equilibrated) matrix A could be poor. This also means that the
 * solution X, estimated condition numbers, and error bounds could be
 * unreliable.
 *
 * @param[in] n     The number of linear equations, i.e., the order of the
 *                  matrix A. n >= 0.
 * @param[in] ncols The number of columns of the matrix A. ncols >= 0.
 * @param[in] A     Single precision array, dimension (lda, n).
 *                  On entry, the N-by-N matrix A.
 * @param[in] lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in] AF    Single precision array, dimension (ldaf, n).
 *                  The factors L and U from the factorization
 *                  A = P*L*U as computed by SGETRF.
 * @param[in] ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @return    The reciprocal pivot growth factor.
 */
f32 sla_gerpvgrw(
    const INT n,
    const INT ncols,
    const f32* restrict A,
    const INT lda,
    const f32* restrict AF,
    const INT ldaf)
{
    INT i, j;
    f32 amax, umax, rpvgrw;

    rpvgrw = 1.0f;

    for (j = 0; j < ncols; j++) {
        amax = 0.0f;
        umax = 0.0f;
        for (i = 0; i < n; i++) {
            amax = fmaxf(fabsf(A[i + j * lda]), amax);
        }
        for (i = 0; i <= j; i++) {
            umax = fmaxf(fabsf(AF[i + j * ldaf]), umax);
        }
        if (umax != 0.0f) {
            rpvgrw = fminf(amax / umax, rpvgrw);
        }
    }
    return rpvgrw;
}
