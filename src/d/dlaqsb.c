/**
 * @file dlaqsb.c
 * @brief DLAQSB scales a symmetric band matrix using scaling factors computed by DPBEQU.
 */

#include "semicolon_lapack_double.h"

#define THRESH 0.1

/**
 * DLAQSB equilibrates a symmetric band matrix A using the scaling
 * factors in the vector S.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular
 *                       - `'L'`: Lower triangular
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in,out] AB     Array of dimension (`ldab`, `n`).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       band matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of A is stored in the j-th
 *                       column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 *                       On exit, the scaled matrix if `equed='Y'`.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[in]     S      Array of dimension `n`.
 *                       The scale factors for A.
 * @param[in]     scond  Ratio of the smallest S(i) to the largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed
 *                       - `'N'`: No equilibration
 *                       - `'Y'`: Equilibration was done, i.e., A has been
 *                         replaced by diag(S) * A * diag(S)
 */
void dlaqsb(
    const char* uplo,
    const INT n,
    const INT kd,
    f64* restrict AB,
    const INT ldab,
    const f64* restrict S,
    const f64 scond,
    const f64 amax,
    char* equed)
{
    const f64 ONE = 1.0;

    INT i, j;
    f64 cj, large, small;

    if (n <= 0) {
        *equed = 'N';
        return;
    }

    small = dlamch("S") / dlamch("P");
    large = ONE / small;

    if (scond >= THRESH && amax >= small && amax <= large) {
        // No equilibration
        *equed = 'N';
    } else {
        // Replace A by diag(S) * A * diag(S)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // Upper triangle of A is stored in band format
            for (j = 0; j < n; j++) {
                cj = S[j];
                for (i = (0 > j - kd ? 0 : j - kd); i <= j; i++) {
                    AB[kd + i - j + j * ldab] = cj * S[i] * AB[kd + i - j + j * ldab];
                }
            }
        } else {
            // Lower triangle of A is stored
            for (j = 0; j < n; j++) {
                cj = S[j];
                for (i = j; i < (n < j + kd + 1 ? n : j + kd + 1); i++) {
                    AB[i - j + j * ldab] = cj * S[i] * AB[i - j + j * ldab];
                }
            }
        }
        *equed = 'Y';
    }
}
