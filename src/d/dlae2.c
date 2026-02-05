/**
 * @file dlae2.c
 * @brief DLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix
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
void dlae2(const double a, const double b, const double c,
           double* rt1, double* rt2)
{
    double sm, df, adf, tb, ab, rt;
    double acmx, acmn;

    /* Compute the eigenvalues */
    sm = a + c;
    df = a - c;
    adf = fabs(df);
    tb = b + b;
    ab = fabs(tb);

    if (fabs(a) > fabs(c)) {
        acmx = a;
        acmn = c;
    } else {
        acmx = c;
        acmn = a;
    }

    if (adf > ab) {
        rt = adf * sqrt(1.0 + (ab / adf) * (ab / adf));
    } else if (adf < ab) {
        rt = ab * sqrt(1.0 + (adf / ab) * (adf / ab));
    } else {
        /* Includes case AB=ADF=0 */
        rt = ab * sqrt(2.0);
    }

    if (sm < 0.0) {
        *rt1 = 0.5 * (sm - rt);

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else if (sm > 0.0) {
        *rt1 = 0.5 * (sm + rt);

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else {
        /* Includes case RT1 = RT2 = 0 */
        *rt1 = 0.5 * rt;
        *rt2 = -0.5 * rt;
    }
}
