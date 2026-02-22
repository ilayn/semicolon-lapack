/**
 * @file slae2.c
 * @brief SLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix
 *    [  A   B  ]
 *    [  B   C  ].
 * On return, RT1 is the eigenvalue of larger absolute value, and RT2
 * is the eigenvalue of smaller absolute value.
 *
 * @param[in]  a    The (1,1) element of the 2-by-2 matrix.
 * @param[in]  b    The (1,2) and (2,1) elements of the 2-by-2 matrix.
 * @param[in]  c    The (2,2) element of the 2-by-2 matrix.
 * @param[out] rt1  The eigenvalue of larger absolute value.
 * @param[out] rt2  The eigenvalue of smaller absolute value.
 */
void slae2(const f32 a, const f32 b, const f32 c,
           f32* rt1, f32* rt2)
{
    f32 sm, df, adf, tb, ab, rt;
    f32 acmx, acmn;

    /* Compute the eigenvalues */
    sm = a + c;
    df = a - c;
    adf = fabsf(df);
    tb = b + b;
    ab = fabsf(tb);

    if (fabsf(a) > fabsf(c)) {
        acmx = a;
        acmn = c;
    } else {
        acmx = c;
        acmn = a;
    }

    if (adf > ab) {
        rt = adf * sqrtf(1.0f + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * sqrtf(1.0f + (adf / ab) * (adf / ab));
    } else {
        /* Includes case AB=ADF=0 */
        rt = ab * sqrtf(2.0f);
    }

    if (sm < 0.0f) {
        *rt1 = 0.5f * (sm - rt);

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else if (sm > 0.0f) {
        *rt1 = 0.5f * (sm + rt);

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else {
        /* Includes case RT1 = RT2 = 0 */
        *rt1 = 0.5f * rt;
        *rt2 = -0.5f * rt;
    }
}
