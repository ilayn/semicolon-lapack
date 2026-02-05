/**
 * @file dlaev2.c
 * @brief DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
 *    [  A   B  ]
 *    [  B   C  ].
 * On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
 * eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
 * eigenvector for RT1, giving the decomposition
 *
 *    [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
 *    [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
 *
 * @param[in]  a    The (1,1) element of the 2-by-2 matrix.
 * @param[in]  b    The (1,2) element of the 2-by-2 matrix.
 * @param[in]  c    The (2,2) element of the 2-by-2 matrix.
 * @param[out] rt1  The eigenvalue of larger absolute value.
 * @param[out] rt2  The eigenvalue of smaller absolute value.
 * @param[out] cs1  The cosine of the rotation.
 * @param[out] sn1  The sine of the rotation.
 *                  The vector (CS1, SN1) is a unit right eigenvector for RT1.
 */
void dlaev2(const double a, const double b, const double c,
            double* rt1, double* rt2, double* cs1, double* sn1)
{
    int sgn1, sgn2;
    double sm, df, adf, tb, ab, rt;
    double acmx, acmn;
    double cs, acs, ct, tn;

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
        sgn1 = -1;

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else if (sm > 0.0) {
        *rt1 = 0.5 * (sm + rt);
        sgn1 = 1;

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else {
        /* Includes case RT1 = RT2 = 0 */
        *rt1 = 0.5 * rt;
        *rt2 = -0.5 * rt;
        sgn1 = 1;
    }

    /* Compute the eigenvector */
    if (df >= 0.0) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }
    acs = fabs(cs);

    if (acs > ab) {
        ct = -tb / cs;
        *sn1 = 1.0 / sqrt(1.0 + ct * ct);
        *cs1 = ct * (*sn1);
    } else {
        if (ab == 0.0) {
            *cs1 = 1.0;
            *sn1 = 0.0;
        } else {
            tn = -cs / tb;
            *cs1 = 1.0 / sqrt(1.0 + tn * tn);
            *sn1 = tn * (*cs1);
        }
    }

    if (sgn1 == sgn2) {
        tn = *cs1;
        *cs1 = -(*sn1);
        *sn1 = tn;
    }
}
