/**
 * @file ssxt1.c
 * @brief SSXT1 computes the difference between two sets of eigenvalues.
 *
 * Port of LAPACK's TESTING/EIG/ssxt1.f to C.
 */

#include "verify.h"
#include <math.h>

/**
 * SSXT1 computes the difference between a set of eigenvalues.
 * The result is returned as the function value.
 *
 * IJOB = 1:   Computes   max { min | D1(i)-D2(j) | }
 *                         i     j
 *
 * IJOB = 2:   Computes   max { min | D1(i)-D2(j) | /
 *                         i     j
 *                              ( ABSTOL + |D1(i)|*ULP ) }
 *
 * @param[in]     ijob    Type of comparison.
 *                        1: absolute difference
 *                        2: relative difference
 * @param[in]     D1      First array of eigenvalues in increasing order, dimension (n1).
 * @param[in]     n1      Length of D1.
 * @param[in]     D2      Second array of eigenvalues in increasing order, dimension (n2).
 * @param[in]     n2      Length of D2.
 * @param[in]     abstol  Absolute tolerance, used as a measure of error.
 * @param[in]     ulp     Machine precision.
 * @param[in]     unfl    Smallest positive number whose reciprocal does not overflow.
 *
 * @return The computed difference between eigenvalue sets.
 */
f32 ssxt1(const int ijob,
             const f32* const restrict D1, const int n1,
             const f32* const restrict D2, const int n2,
             const f32 abstol, const f32 ulp, const f32 unfl)
{
    const f32 ZERO = 0.0f;
    f32 temp1, temp2;
    int i, j;

    temp1 = ZERO;

    j = 0;
    for (i = 0; i < n1; i++) {
        /* Advance j until D2[j] >= D1[i] or j reaches end */
        while (D2[j] < D1[i] && j < n2 - 1) {
            j++;
        }

        if (j == 0) {
            /* Only one candidate: D2[0] */
            temp2 = fabsf(D2[0] - D1[i]);
            if (ijob == 2) {
                f32 denom = fmaxf(unfl, abstol + ulp * fabsf(D1[i]));
                temp2 = temp2 / denom;
            }
        } else {
            /* Two candidates: D2[j] and D2[j-1] */
            temp2 = fminf(fabsf(D2[j] - D1[i]), fabsf(D1[i] - D2[j - 1]));
            if (ijob == 2) {
                f32 denom = fmaxf(unfl, abstol + ulp * fabsf(D1[i]));
                temp2 = temp2 / denom;
            }
        }

        if (temp2 > temp1) {
            temp1 = temp2;
        }
    }

    return temp1;
}
