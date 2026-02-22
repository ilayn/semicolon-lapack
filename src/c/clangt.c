/**
 * @file clangt.c
 * @brief CLANGT returns the value of the 1-norm, Frobenius norm, infinity-norm,
 *        or the largest absolute value of any element of a general tridiagonal matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLANGT returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex tridiagonal matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (maximum column sum)
 *                  = 'I' or 'i': normI(A) (maximum row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] n     The order of the matrix A. n >= 0. When n = 0, CLANGT is
 *                  set to zero.
 * @param[in] DL    The (n-1) sub-diagonal elements of A. Array of dimension (n-1).
 * @param[in] D     The diagonal elements of A. Array of dimension (n).
 * @param[in] DU    The (n-1) super-diagonal elements of A. Array of dimension (n-1).
 *
 * @return The computed norm value.
 */
f32 clangt(
    const char* norm,
    const INT n,
    const c64* restrict DL,
    const c64* restrict D,
    const c64* restrict DU)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    f32 anorm, scale, sum, temp;
    INT i;

    if (n <= 0) {
        anorm = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        anorm = cabsf(D[n - 1]);
        for (i = 0; i < n - 1; i++) {
            temp = cabsf(DL[i]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            temp = cabsf(D[i]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            temp = cabsf(DU[i]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find norm1(A) - maximum column sum */
        if (n == 1) {
            anorm = cabsf(D[0]);
        } else {
            /* First column: D[0] + DL[0] */
            anorm = cabsf(D[0]) + cabsf(DL[0]);
            /* Last column: DU[n-2] + D[n-1] */
            temp = cabsf(D[n - 1]) + cabsf(DU[n - 2]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            /* Middle columns: DU[i-1] + D[i] + DL[i] */
            for (i = 1; i < n - 1; i++) {
                temp = cabsf(D[i]) + cabsf(DL[i]) + cabsf(DU[i - 1]);
                if (anorm < temp || isnan(temp)) {
                    anorm = temp;
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) - maximum row sum */
        if (n == 1) {
            anorm = cabsf(D[0]);
        } else {
            /* First row: D[0] + DU[0] */
            anorm = cabsf(D[0]) + cabsf(DU[0]);
            /* Last row: DL[n-2] + D[n-1] */
            temp = cabsf(D[n - 1]) + cabsf(DL[n - 2]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            /* Middle rows: DL[i-1] + D[i] + DU[i] */
            for (i = 1; i < n - 1; i++) {
                temp = cabsf(D[i]) + cabsf(DU[i]) + cabsf(DL[i - 1]);
                if (anorm < temp || isnan(temp)) {
                    anorm = temp;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) - Frobenius norm */
        scale = ZERO;
        sum = ONE;
        classq(n, D, 1, &scale, &sum);
        if (n > 1) {
            classq(n - 1, DL, 1, &scale, &sum);
            classq(n - 1, DU, 1, &scale, &sum);
        }
        anorm = scale * sqrtf(sum);
    } else {
        /* Default to zero for unrecognized norm */
        anorm = ZERO;
    }

    return anorm;
}
