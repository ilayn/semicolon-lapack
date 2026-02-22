/**
 * @file clarfg.c
 * @brief CLARFG generates an elementary reflector (Householder matrix).
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLARFG generates a complex elementary reflector H of order n, such
 * that
 *
 *       H**H * ( alpha ) = ( beta ),   H**H * H = I.
 *              (   x   )   (   0  )
 *
 * where alpha and beta are scalars, with beta real, and x is an
 * (n-1)-element complex vector. H is represented in the form
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
 * Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
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
void clarfg(const INT n, c64* alpha, c64* x,
            const INT incx, c64* tau)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT j, knt;
    f32 alphi, alphr, beta, rsafmn, safmin, xnorm;

    if (n <= 0) {
        *tau = CMPLXF(ZERO, ZERO);
        return;
    }

    xnorm = cblas_scnrm2(n - 1, x, incx);
    alphr = crealf(*alpha);
    alphi = cimagf(*alpha);

    if (xnorm == ZERO && alphi == ZERO) {

        /* H  =  I */

        *tau = CMPLXF(ZERO, ZERO);
    } else {

        /* general case */

        beta = -copysignf(slapy3(alphr, alphi, xnorm), alphr);
        safmin = FLT_MIN / FLT_EPSILON;
        rsafmn = ONE / safmin;

        knt = 0;
        if (fabsf(beta) < safmin) {

            /* XNORM, BETA may be inaccurate; scale X and recompute them */

            for (knt = 0; knt < 20; knt++) {
                cblas_csscal(n - 1, rsafmn, x, incx);
                beta = beta * rsafmn;
                alphi = alphi * rsafmn;
                alphr = alphr * rsafmn;
                if (fabsf(beta) >= safmin) {
                    break;
                }
            }
            knt++;  /* Convert from 0-based loop count to iteration count */

            /* New BETA is at most 1, at least SAFMIN */

            xnorm = cblas_scnrm2(n - 1, x, incx);
            *alpha = CMPLXF(alphr, alphi);
            beta = -copysignf(slapy3(alphr, alphi, xnorm), alphr);
        }
        *tau = CMPLXF((beta - alphr) / beta, -alphi / beta);
        *alpha = cladiv(CMPLXF(ONE, ZERO), *alpha - beta);
        cblas_cscal(n - 1, alpha, x, incx);

        /* If ALPHA is subnormal, it may lose relative accuracy */

        for (j = 0; j < knt; j++) {
            beta = beta * safmin;
        }
        *alpha = CMPLXF(beta, ZERO);
    }
}
