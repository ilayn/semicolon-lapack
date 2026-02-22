/**
 * @file zlarfg.c
 * @brief ZLARFG generates an elementary reflector (Householder matrix).
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARFG generates a complex elementary reflector H of order n, such
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
 * @param[in,out] x      Double complex array, dimension (1+(n-2)*abs(incx)).
 *                       On entry, the vector x.
 *                       On exit, it is overwritten with the vector v.
 * @param[in]     incx   The increment between elements of x. incx > 0.
 * @param[out]    tau    The value tau.
 */
void zlarfg(const INT n, c128* alpha, c128* x,
            const INT incx, c128* tau)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT j, knt;
    f64 alphi, alphr, beta, rsafmn, safmin, xnorm;

    if (n <= 0) {
        *tau = CMPLX(ZERO, ZERO);
        return;
    }

    xnorm = cblas_dznrm2(n - 1, x, incx);
    alphr = creal(*alpha);
    alphi = cimag(*alpha);

    if (xnorm == ZERO && alphi == ZERO) {

        /* H  =  I */

        *tau = CMPLX(ZERO, ZERO);
    } else {

        /* general case */

        beta = -copysign(dlapy3(alphr, alphi, xnorm), alphr);
        safmin = DBL_MIN / DBL_EPSILON;
        rsafmn = ONE / safmin;

        knt = 0;
        if (fabs(beta) < safmin) {

            /* XNORM, BETA may be inaccurate; scale X and recompute them */

            for (knt = 0; knt < 20; knt++) {
                cblas_zdscal(n - 1, rsafmn, x, incx);
                beta = beta * rsafmn;
                alphi = alphi * rsafmn;
                alphr = alphr * rsafmn;
                if (fabs(beta) >= safmin) {
                    break;
                }
            }
            knt++;  /* Convert from 0-based loop count to iteration count */

            /* New BETA is at most 1, at least SAFMIN */

            xnorm = cblas_dznrm2(n - 1, x, incx);
            *alpha = CMPLX(alphr, alphi);
            beta = -copysign(dlapy3(alphr, alphi, xnorm), alphr);
        }
        *tau = CMPLX((beta - alphr) / beta, -alphi / beta);
        *alpha = zladiv(CMPLX(ONE, ZERO), *alpha - beta);
        cblas_zscal(n - 1, alpha, x, incx);

        /* If ALPHA is subnormal, it may lose relative accuracy */

        for (j = 0; j < knt; j++) {
            beta = beta * safmin;
        }
        *alpha = CMPLX(beta, ZERO);
    }
}
