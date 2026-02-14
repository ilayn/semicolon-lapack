/**
 * @file slangb.c
 * @brief SLANGB returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a general band matrix.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLANGB returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of an
 * n by n band matrix A, with kl sub-diagonals and ku super-diagonals.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] n     The order of the matrix A. n >= 0.
 *                  When n = 0, slangb returns zero.
 * @param[in] kl    The number of sub-diagonals of the matrix A. kl >= 0.
 * @param[in] ku    The number of super-diagonals of the matrix A. ku >= 0.
 * @param[in] AB    Double precision array, dimension (ldab, n).
 *                  The band matrix A, stored in rows 0 to kl+ku (0-based).
 *                  The j-th column of A is stored in the j-th column of
 *                  the array AB as follows:
 *                  AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku) <= i <= min(n-1,j+kl).
 * @param[in] ldab  The leading dimension of the array AB. ldab >= kl+ku+1.
 * @param[out] work Double precision array, dimension (max(1, lwork)).
 *                  where lwork >= n when norm = 'I'; otherwise, work is
 *                  not referenced.
 *
 * @return The computed norm value.
 */
f32 slangb(
    const char* norm,
    const int n,
    const int kl,
    const int ku,
    const f32* restrict AB,
    const int ldab,
    f32* restrict work)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int i, j, k, l;
    f32 scale, sum, value, temp;

    /* Quick return if possible */
    if (n == 0) {
        return ZERO;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        value = ZERO;
        for (j = 0; j < n; j++) {
            /*
             * Fortran (1-based): DO I = MAX(KU+2-J, 1), MIN(N+KU+1-J, KL+KU+1)
             * C (0-based): j_c = j_f - 1, i_c = i_f - 1
             *   i_start = max(ku+2-(j+1), 1) - 1 = max(ku-j, 0)
             *   i_end   = min(n+ku+1-(j+1), kl+ku+1) - 1 = min(n-1+ku-j, kl+ku)
             *
             * These are the row indices in band storage for column j.
             */
            int i_start = (ku - j > 0) ? ku - j : 0;
            int i_end = (n - 1 + ku - j < kl + ku) ? n - 1 + ku - j : kl + ku;
            for (i = i_start; i <= i_end; i++) {
                temp = fabsf(AB[i + j * ldab]);
                if (value < temp || isnan(temp)) {
                    value = temp;
                }
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find norm1(A) - maximum column sum */
        value = ZERO;
        for (j = 0; j < n; j++) {
            sum = ZERO;
            int i_start = (ku - j > 0) ? ku - j : 0;
            int i_end = (n - 1 + ku - j < kl + ku) ? n - 1 + ku - j : kl + ku;
            for (i = i_start; i <= i_end; i++) {
                sum += fabsf(AB[i + j * ldab]);
            }
            if (value < sum || isnan(sum)) {
                value = sum;
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) - maximum row sum */
        for (i = 0; i < n; i++) {
            work[i] = ZERO;
        }
        for (j = 0; j < n; j++) {
            /*
             * For infinity norm, we iterate over actual matrix rows.
             * k is the offset in band storage: AB[k + i, j] = A(i, j)
             * In 0-based: k = ku - j, so AB[ku - j + i + j*ldab] = A(i,j)
             */
            k = ku - j;
            int row_start = (j - ku > 0) ? j - ku : 0;
            int row_end = (j + kl < n - 1) ? j + kl : n - 1;
            for (i = row_start; i <= row_end; i++) {
                work[i] += fabsf(AB[k + i + j * ldab]);
            }
        }
        value = ZERO;
        for (i = 0; i < n; i++) {
            temp = work[i];
            if (value < temp || isnan(temp)) {
                value = temp;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) - Frobenius norm using slassq */
        scale = ZERO;
        sum = ONE;
        for (j = 0; j < n; j++) {
            /*
             * l is the first row index in the original matrix for column j
             * In 0-based: l = max(0, j - ku)
             * k is the corresponding row in band storage: k = ku - j + l
             * Number of elements: min(n-1, j+kl) - l + 1
             */
            l = (j - ku > 0) ? j - ku : 0;
            k = ku - j + l;
            int count = ((j + kl < n - 1) ? j + kl : n - 1) - l + 1;
            slassq(count, &AB[k + j * ldab], 1, &scale, &sum);
        }
        value = scale * sqrtf(sum);
    } else {
        value = ZERO;
    }

    return value;
}
