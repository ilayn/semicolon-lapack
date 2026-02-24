/**
 * @file sget53.c
 * @brief SGET53 checks the generalized eigenvalues computed by SLAG2.
 */

#include "verify.h"
#include <math.h>

/**
 * SGET53 checks the generalized eigenvalues computed by SLAG2.
 *
 * The basic test for an eigenvalue is:
 *
 *                              | det( s A - w B ) |
 *     RESULT =  ---------------------------------------------------
 *               ulp max( s norm(A), |w| norm(B) )*norm( s A - w B )
 *
 * Two "safety checks" are performed:
 *
 * (1)  ulp*max( s*norm(A), |w|*norm(B) )  must be at least
 *      safe_minimum.  This insures that the test performed is
 *      not essentially  det(0*A + 0*B)=0.
 *
 * (2)  s*norm(A) + |w|*norm(B) must be less than 1/safe_minimum.
 *      This insures that  s*A - w*B  will not overflow.
 *
 * If these tests are not passed, then  s  and  w  are scaled and
 * tested anyway, if this is possible.
 *
 * @param[in]     A       The 2x2 matrix A, dimension (lda, 2).
 * @param[in]     lda     The leading dimension of A. lda >= 2.
 * @param[in]     B       The 2x2 upper-triangular matrix B, dimension (ldb, 2).
 * @param[in]     ldb     The leading dimension of B. ldb >= 2.
 * @param[in]     scale   The "scale factor" s in the formula s A - w B.
 *                        It is assumed to be non-negative.
 * @param[in]     wr      The real part of the eigenvalue w.
 * @param[in]     wi      The imaginary part of the eigenvalue w.
 * @param[out]    result  The computed test value.
 * @param[out]    info    = 0: The input data pass the "safety checks".
 *                        = 1: s*norm(A) + |w|*norm(B) > 1/safe_minimum.
 *                        = 2: ulp*max( s*norm(A), |w|*norm(B) ) < safe_minimum
 *                        = 3: same as 2, but s and w could not be scaled.
 */
void sget53(const f32* A, const INT lda, const f32* B, const INT ldb,
            const f32 scale, const f32 wr, const f32 wi,
            f32* result, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    *info = 0;
    *result = ZERO;
    f32 scales = scale;
    f32 wrs = wr;
    f32 wis = wi;

    f32 safmin = slamch("Safe minimum");
    f32 ulp = slamch("Epsilon") * slamch("Base");
    f32 absw = fabsf(wrs) + fabsf(wis);
    f32 anorm = fmaxf(fabsf(A[0]) + fabsf(A[1]),
                fmaxf(fabsf(A[lda]) + fabsf(A[1 + lda]), safmin));
    f32 bnorm = fmaxf(fabsf(B[0]),
                fmaxf(fabsf(B[ldb]) + fabsf(B[1 + ldb]), safmin));

    f32 temp = (safmin * bnorm) * absw + (safmin * anorm) * scales;
    if (temp >= ONE) {
        *info = 1;
        temp = ONE / temp;
        scales = scales * temp;
        wrs = wrs * temp;
        wis = wis * temp;
        absw = fabsf(wrs) + fabsf(wis);
    }
    f32 s1 = fmaxf(ulp * fmaxf(scales * anorm, absw * bnorm),
                  safmin * fmaxf(scales, absw));

    if (s1 < safmin) {
        *info = 2;
        if (scales < safmin && absw < safmin) {
            *info = 3;
            *result = ONE / ulp;
            return;
        }

        temp = ONE / fmaxf(scales * anorm + absw * bnorm, safmin);
        scales = scales * temp;
        wrs = wrs * temp;
        wis = wis * temp;
        absw = fabsf(wrs) + fabsf(wis);
        s1 = fmaxf(ulp * fmaxf(scales * anorm, absw * bnorm),
                  safmin * fmaxf(scales, absw));
        if (s1 < safmin) {
            *info = 3;
            *result = ONE / ulp;
            return;
        }
    }

    f32 cr11 = scales * A[0]         - wrs * B[0];
    f32 ci11 =                        -wis * B[0];
    f32 cr21 = scales * A[1];
    f32 cr12 = scales * A[lda]       - wrs * B[ldb];
    f32 ci12 =                        -wis * B[ldb];
    f32 cr22 = scales * A[1 + lda]   - wrs * B[1 + ldb];
    f32 ci22 =                        -wis * B[1 + ldb];

    f32 cnorm = fmaxf(fabsf(cr11) + fabsf(ci11) + fabsf(cr21),
                fmaxf(fabsf(cr12) + fabsf(ci12) + fabsf(cr22) + fabsf(ci22), safmin));
    f32 cscale = ONE / sqrtf(cnorm);
    f32 detr = (cscale * cr11) * (cscale * cr22) -
               (cscale * ci11) * (cscale * ci22) -
               (cscale * cr12) * (cscale * cr21);
    f32 deti = (cscale * cr11) * (cscale * ci22) +
               (cscale * ci11) * (cscale * cr22) -
               (cscale * ci12) * (cscale * cr21);
    f32 sigmin = fabsf(detr) + fabsf(deti);
    *result = sigmin / s1;
}
