/**
 * @file zlaqhb.c
 * @brief ZLAQHB scales a Hermitian band matrix using scaling factors computed by ZPBEQU.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

#define THRESH 0.1

/**
 * ZLAQHB equilibrates a Hermitian band matrix A using the scaling
 * factors in the vector S.
 *
 * @param[in]     uplo   = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     The banded matrix A. On exit, the scaled matrix if EQUED='Y'.
 *                       Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in]     S      The scale factors for A. Array of dimension (n).
 * @param[in]     scond  Ratio of smallest to largest S(i).
 * @param[in]     amax   Absolute value of largest matrix entry.
 * @param[out]    equed  = 'N': No equilibration
 *                       = 'Y': Equilibration was done
 */
void zlaqhb(
    const char* uplo,
    const INT n,
    const INT kd,
    c128* restrict AB,
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
        *equed = 'N';
    } else {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                cj = S[j];
                for (i = (0 > j - kd ? 0 : j - kd); i < j; i++) {
                    AB[kd + i - j + j * ldab] = cj * S[i] * AB[kd + i - j + j * ldab];
                }
                AB[kd + j * ldab] = cj * cj * creal(AB[kd + j * ldab]);
            }
        } else {
            for (j = 0; j < n; j++) {
                cj = S[j];
                AB[j * ldab] = cj * cj * creal(AB[j * ldab]);
                for (i = j + 1; i < (n < j + kd + 1 ? n : j + kd + 1); i++) {
                    AB[i - j + j * ldab] = cj * S[i] * AB[i - j + j * ldab];
                }
            }
        }
        *equed = 'Y';
    }
}
