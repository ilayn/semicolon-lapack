/**
 * @file clarfgp.c
 * @brief CLARFGP generates an elementary reflector (Householder matrix)
 *        with non-negative beta.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLARFGP generates a complex elementary reflector H of order n, such
 * that
 *
 *       H**H * ( alpha ) = ( beta ),   H**H * H = I.
 *              (   x   )   (   0  )
 *
 * where alpha and beta are scalars, beta is real and non-negative, and
 * x is an (n-1)-element complex vector.  H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v**H ) ,
 *                     ( v )
 *
 * where tau is a complex scalar and v is a complex (n-1)-element
 * vector. Note that H is not hermitian.
 *
 * If the elements of x are all zero and alpha is real, then tau = 0
 * and H is taken to be the unit matrix.
 *
 * @param[in]     n      The order of the elementary reflector.
 * @param[in,out] alpha  On entry, the value alpha.
 *                       On exit, it is overwritten with the value beta.
 * @param[in,out] x      Single complex array, dimension (1+(n-2)*abs(incx)).
 *                       On entry, the vector x.
 *                       On exit, it is overwritten with the vector v.
 * @param[in]     incx   The increment between elements of x. incx > 0.
 * @param[out]    tau    The value tau.
 */
void clarfgp(const INT n, c64* alpha, c64* restrict x,
             const INT incx, c64* tau)
{
    const f32 TWO = 2.0f;
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT j, knt;
    f32 alphi, alphr, beta, bignum, eps, smlnum, xnorm;
    c64 savealpha;

    if (n <= 0) {
        *tau = CMPLXF(ZERO, ZERO);
        return;
    }

    eps = FLT_EPSILON;
    xnorm = cblas_scnrm2(n - 1, x, incx);
    alphr = crealf(*alpha);
    alphi = cimagf(*alpha);

    if (xnorm <= eps * cabsf(*alpha) && alphi == ZERO) {

        /*
         * H  =  [1-alpha/abs(alpha) 0; 0 I], sign chosen so ALPHA >= 0.
         */

        if (alphr >= ZERO) {
            /* When TAU.eq.ZERO, the vector is special-cased to be
             * all zeros in the application routines.  We do not need
             * to clear it. */
            *tau = CMPLXF(ZERO, ZERO);
        } else {
            /* However, the application routines rely on explicit
             * zero checks when TAU.ne.ZERO, and we must clear X. */
            *tau = CMPLXF(TWO, ZERO);
            for (j = 0; j < n - 1; j++) {
                x[j * incx] = CMPLXF(ZERO, ZERO);
            }
            *alpha = -*alpha;
        }
    } else {

        /*
         * general case
         */

        beta = copysignf(slapy3(alphr, alphi, xnorm), alphr);
        smlnum = FLT_MIN / FLT_EPSILON;
        bignum = ONE / smlnum;

        knt = 0;
        if (fabsf(beta) < smlnum) {

            /*
             * XNORM, BETA may be inaccurate; scale X and recompute them
             */

            while (fabsf(beta) < smlnum && knt < 20) {
                knt++;
                cblas_csscal(n - 1, bignum, x, incx);
                beta = beta * bignum;
                alphi = alphi * bignum;
                alphr = alphr * bignum;
            }

            /* New BETA is at most 1, at least SMLNUM */

            xnorm = cblas_scnrm2(n - 1, x, incx);
            *alpha = CMPLXF(alphr, alphi);
            beta = copysignf(slapy3(alphr, alphi, xnorm), alphr);
        }
        savealpha = *alpha;
        *alpha = *alpha + beta;
        if (beta < ZERO) {
            beta = -beta;
            *tau = -*alpha / beta;
        } else {
            alphr = alphi * (alphi / crealf(*alpha));
            alphr = alphr + xnorm * (xnorm / crealf(*alpha));
            *tau = CMPLXF(alphr / beta, -alphi / beta);
            *alpha = CMPLXF(-alphr, alphi);
        }
        *alpha = cladiv(CMPLXF(ONE, ZERO), *alpha);

        if (cabsf(*tau) <= smlnum) {

            /*
             * In the case where the computed TAU ends up being a
             * denormalized number, it loses relative accuracy. This is
             * a BIG problem. Solution: flush TAU to ZERO (or TWO or
             * whatever makes a nonnegative real number for BETA).
             *
             * (Bug report provided by Pat Quillen from MathWorks
             *  on Jul 29, 2009.)
             * (Thanks Pat. Thanks MathWorks.)
             */

            alphr = crealf(savealpha);
            alphi = cimagf(savealpha);
            if (alphi == ZERO) {
                if (alphr >= ZERO) {
                    *tau = CMPLXF(ZERO, ZERO);
                } else {
                    *tau = CMPLXF(TWO, ZERO);
                    for (j = 0; j < n - 1; j++) {
                        x[j * incx] = CMPLXF(ZERO, ZERO);
                    }
                    beta = crealf(-savealpha);
                }
            } else {
                xnorm = slapy2(alphr, alphi);
                *tau = CMPLXF(ONE - alphr / xnorm, -alphi / xnorm);
                for (j = 0; j < n - 1; j++) {
                    x[j * incx] = CMPLXF(ZERO, ZERO);
                }
                beta = xnorm;
            }

        } else {

            /*
             * This is the general case.
             */

            cblas_cscal(n - 1, alpha, x, incx);

        }

        /* If BETA is subnormal, it may lose relative accuracy */

        for (j = 0; j < knt; j++) {
            beta = beta * smlnum;
        }
        *alpha = CMPLXF(beta, ZERO);
    }
}
