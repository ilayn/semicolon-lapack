/**
 * @file slaev2.c
 * @brief SLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
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
void slaev2(const float a, const float b, const float c,
            float* rt1, float* rt2, float* cs1, float* sn1)
{
    int sgn1, sgn2;
    float sm, df, adf, tb, ab, rt;
    float acmx, acmn;
    float cs, acs, ct, tn;

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
        sgn1 = -1;

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else if (sm > 0.0f) {
        *rt1 = 0.5f * (sm + rt);
        sgn1 = 1;

        /* Order of execution important.
         * To get fully accurate smaller eigenvalue,
         * next line needs to be executed in higher precision. */
        *rt2 = (acmx / (*rt1)) * acmn - (b / (*rt1)) * b;
    } else {
        /* Includes case RT1 = RT2 = 0 */
        *rt1 = 0.5f * rt;
        *rt2 = -0.5f * rt;
        sgn1 = 1;
    }

    /* Compute the eigenvector */
    if (df >= 0.0f) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }
    acs = fabsf(cs);

    if (acs > ab) {
        ct = -tb / cs;
        *sn1 = 1.0f / sqrtf(1.0f + ct * ct);
        *cs1 = ct * (*sn1);
    } else {
        if (ab == 0.0f) {
            *cs1 = 1.0f;
            *sn1 = 0.0f;
        } else {
            tn = -cs / tb;
            *cs1 = 1.0f / sqrtf(1.0f + tn * tn);
            *sn1 = tn * (*cs1);
        }
    }

    if (sgn1 == sgn2) {
        tn = *cs1;
        *cs1 = -(*sn1);
        *sn1 = tn;
    }
}
