/**
 * @file slangt.c
 * @brief SLANGT returns the value of the 1-norm, Frobenius norm, infinity-norm,
 *        or the largest absolute value of any element of a general tridiagonal matrix.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLANGT returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real tridiagonal matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (maximum column sum)
 *                  = 'I' or 'i': normI(A) (maximum row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] n     The order of the matrix A. n >= 0. When n = 0, SLANGT is
 *                  set to zero.
 * @param[in] DL    The (n-1) sub-diagonal elements of A. Array of dimension (n-1).
 * @param[in] D     The diagonal elements of A. Array of dimension (n).
 * @param[in] DU    The (n-1) super-diagonal elements of A. Array of dimension (n-1).
 *
 * @return The computed norm value.
 */
f32 slangt(
    const char* norm,
    const int n,
    const f32 * const restrict DL,
    const f32 * const restrict D,
    const f32 * const restrict DU)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    f32 anorm, scale, sum, temp;
    int i;

    if (n <= 0) {
        anorm = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        anorm = fabsf(D[n - 1]);
        for (i = 0; i < n - 1; i++) {
            temp = fabsf(DL[i]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            temp = fabsf(D[i]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            temp = fabsf(DU[i]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find norm1(A) - maximum column sum */
        if (n == 1) {
            anorm = fabsf(D[0]);
        } else {
            /* First column: D[0] + DL[0] */
            anorm = fabsf(D[0]) + fabsf(DL[0]);
            /* Last column: DU[n-2] + D[n-1] */
            temp = fabsf(D[n - 1]) + fabsf(DU[n - 2]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            /* Middle columns: DU[i-1] + D[i] + DL[i] */
            for (i = 1; i < n - 1; i++) {
                temp = fabsf(D[i]) + fabsf(DL[i]) + fabsf(DU[i - 1]);
                if (anorm < temp || isnan(temp)) {
                    anorm = temp;
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) - maximum row sum */
        if (n == 1) {
            anorm = fabsf(D[0]);
        } else {
            /* First row: D[0] + DU[0] */
            anorm = fabsf(D[0]) + fabsf(DU[0]);
            /* Last row: DL[n-2] + D[n-1] */
            temp = fabsf(D[n - 1]) + fabsf(DL[n - 2]);
            if (anorm < temp || isnan(temp)) {
                anorm = temp;
            }
            /* Middle rows: DL[i-1] + D[i] + DU[i] */
            for (i = 1; i < n - 1; i++) {
                temp = fabsf(D[i]) + fabsf(DU[i]) + fabsf(DL[i - 1]);
                if (anorm < temp || isnan(temp)) {
                    anorm = temp;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) - Frobenius norm */
        scale = ZERO;
        sum = ONE;
        slassq(n, D, 1, &scale, &sum);
        if (n > 1) {
            slassq(n - 1, DL, 1, &scale, &sum);
            slassq(n - 1, DU, 1, &scale, &sum);
        }
        anorm = scale * sqrtf(sum);
    } else {
        /* Default to zero for unrecognized norm */
        anorm = ZERO;
    }

    return anorm;
}
