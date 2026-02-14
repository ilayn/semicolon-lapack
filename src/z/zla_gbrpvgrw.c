/**
 * @file zla_gbrpvgrw.c
 * @brief ZLA_GBRPVGRW computes the reciprocal pivot growth factor
 *        norm(A)/norm(U) for a general banded matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLA_GBRPVGRW computes the reciprocal pivot growth factor
 * norm(A)/norm(U). The "max absolute element" norm is used. If this is
 * much less than 1, the stability of the LU factorization of the
 * (equilibrated) matrix A could be poor. This also means that the
 * solution X, estimated condition numbers, and error bounds could be
 * unreliable.
 *
 * @param[in] n      The number of linear equations, i.e., the order of the
 *                   matrix A. n >= 0.
 * @param[in] kl     The number of subdiagonals within the band of A. kl >= 0.
 * @param[in] ku     The number of superdiagonals within the band of A. ku >= 0.
 * @param[in] ncols  The number of columns of the matrix A. ncols >= 0.
 * @param[in] AB     Double complex array, dimension (ldab, n).
 *                   On entry, the matrix A in band storage, in rows 0 to kl+ku
 *                   (0-based). The j-th column of A is stored in the j-th
 *                   column of the array AB as follows:
 *                   AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku) <= i <= min(n-1,j+kl)
 * @param[in] ldab   The leading dimension of the array AB. ldab >= kl+ku+1.
 * @param[in] AFB    Double complex array, dimension (ldafb, n).
 *                   Details of the LU factorization of the band matrix A, as
 *                   computed by ZGBTRF. U is stored as an upper triangular
 *                   band matrix with kl+ku superdiagonals in rows 0 to kl+ku
 *                   (0-based), and the multipliers used during the factorization
 *                   are stored in rows kl+ku+1 to 2*kl+ku.
 * @param[in] ldafb  The leading dimension of the array AFB. ldafb >= 2*kl+ku+1.
 *
 * @return The reciprocal pivot growth factor.
 */
f64 zla_gbrpvgrw(
    const int n,
    const int kl,
    const int ku,
    const int ncols,
    const c128* restrict AB,
    const int ldab,
    const c128* restrict AFB,
    const int ldafb)
{
    int i, j, kd;
    f64 amax, umax, rpvgrw;

    rpvgrw = 1.0;

    kd = ku;
    for (j = 0; j < ncols; j++) {
        amax = 0.0;
        umax = 0.0;
        for (i = (j - ku > 0 ? j - ku : 0); i < (j + kl + 1 < n ? j + kl + 1 : n); i++) {
            amax = (cabs1(AB[kd + i - j + j * ldab]) > amax ?
                    cabs1(AB[kd + i - j + j * ldab]) : amax);
        }
        for (i = (j - ku > 0 ? j - ku : 0); i <= j; i++) {
            umax = (cabs1(AFB[kd + i - j + j * ldafb]) > umax ?
                    cabs1(AFB[kd + i - j + j * ldafb]) : umax);
        }
        if (umax != 0.0) {
            rpvgrw = (amax / umax < rpvgrw ? amax / umax : rpvgrw);
        }
    }

    return rpvgrw;
}
