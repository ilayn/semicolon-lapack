/**
 * @file dget53.c
 * @brief DGET53 checks the generalized eigenvalues computed by DLAG2.
 */

#include "verify.h"
#include <math.h>

/**
 * DGET53 checks the generalized eigenvalues computed by DLAG2.
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
void dget53(const f64* A, const INT lda, const f64* B, const INT ldb,
            const f64 scale, const f64 wr, const f64 wi,
            f64* result, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    *result = ZERO;
    f64 scales = scale;
    f64 wrs = wr;
    f64 wis = wi;

    f64 safmin = dlamch("Safe minimum");
    f64 ulp = dlamch("Epsilon") * dlamch("Base");
    f64 absw = fabs(wrs) + fabs(wis);
    f64 anorm = fmax(fabs(A[0]) + fabs(A[1]),
                fmax(fabs(A[lda]) + fabs(A[1 + lda]), safmin));
    f64 bnorm = fmax(fabs(B[0]),
                fmax(fabs(B[ldb]) + fabs(B[1 + ldb]), safmin));

    f64 temp = (safmin * bnorm) * absw + (safmin * anorm) * scales;
    if (temp >= ONE) {
        *info = 1;
        temp = ONE / temp;
        scales = scales * temp;
        wrs = wrs * temp;
        wis = wis * temp;
        absw = fabs(wrs) + fabs(wis);
    }
    f64 s1 = fmax(ulp * fmax(scales * anorm, absw * bnorm),
                  safmin * fmax(scales, absw));

    if (s1 < safmin) {
        *info = 2;
        if (scales < safmin && absw < safmin) {
            *info = 3;
            *result = ONE / ulp;
            return;
        }

        temp = ONE / fmax(scales * anorm + absw * bnorm, safmin);
        scales = scales * temp;
        wrs = wrs * temp;
        wis = wis * temp;
        absw = fabs(wrs) + fabs(wis);
        s1 = fmax(ulp * fmax(scales * anorm, absw * bnorm),
                  safmin * fmax(scales, absw));
        if (s1 < safmin) {
            *info = 3;
            *result = ONE / ulp;
            return;
        }
    }

    f64 cr11 = scales * A[0]         - wrs * B[0];
    f64 ci11 =                        -wis * B[0];
    f64 cr21 = scales * A[1];
    f64 cr12 = scales * A[lda]       - wrs * B[ldb];
    f64 ci12 =                        -wis * B[ldb];
    f64 cr22 = scales * A[1 + lda]   - wrs * B[1 + ldb];
    f64 ci22 =                        -wis * B[1 + ldb];

    f64 cnorm = fmax(fabs(cr11) + fabs(ci11) + fabs(cr21),
                fmax(fabs(cr12) + fabs(ci12) + fabs(cr22) + fabs(ci22), safmin));
    f64 cscale = ONE / sqrt(cnorm);
    f64 detr = (cscale * cr11) * (cscale * cr22) -
               (cscale * ci11) * (cscale * ci22) -
               (cscale * cr12) * (cscale * cr21);
    f64 deti = (cscale * cr11) * (cscale * ci22) +
               (cscale * ci11) * (cscale * cr22) -
               (cscale * ci12) * (cscale * cr21);
    f64 sigmin = fabs(detr) + fabs(deti);
    *result = sigmin / s1;
}
