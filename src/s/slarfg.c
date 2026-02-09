/**
 * @file slarfg.c
 * @brief SLARFG generates an elementary reflector (Householder matrix).
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLARFG generates a real elementary reflector H of order n, such that
 *
 *       H * ( alpha ) = ( beta ),   H**T * H = I.
 *           (   x   )   (   0  )
 *
 * where alpha and beta are scalars, and x is an (n-1)-element real
 * vector. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v**T ),
 *                     ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element vector.
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit matrix.
 *
 * Otherwise  1 <= tau <= 2.
 *
 * @param[in]     n      The order of the elementary reflector.
 * @param[in,out] alpha  On entry, the value alpha.
 *                       On exit, it is overwritten with the value beta.
 * @param[in,out] x      Double precision array, dimension (1+(n-2)*abs(incx)).
 *                       On entry, the vector x.
 *                       On exit, it is overwritten with the vector v.
 * @param[in]     incx   The increment between elements of x. incx > 0.
 * @param[out]    tau    The value tau.
 */
void slarfg(const int n, float *alpha, float * const x,
            const int incx, float *tau)
{
    const float ONE = 1.0f;
    float xnorm, beta, safmin, rsafmn;
    int knt, j;

    if (n <= 1) {
        *tau = 0.0f;
        return;
    }

    xnorm = cblas_snrm2(n - 1, x, incx);

    if (xnorm == 0.0f) {
        /* H = I */
        *tau = 0.0f;
    } else {
        /* General case */
        beta = -copysignf(slapy2(*alpha, xnorm), *alpha);
        safmin = FLT_MIN / FLT_EPSILON;
        knt = 0;

        if (fabsf(beta) < safmin) {
            /* XNORM, BETA may be inaccurate; scale X and recompute them */
            rsafmn = ONE / safmin;
            for (knt = 0; knt < 20; knt++) {
                cblas_sscal(n - 1, rsafmn, x, incx);
                beta = beta * rsafmn;
                *alpha = (*alpha) * rsafmn;
                if (fabsf(beta) >= safmin) {
                    break;
                }
            }
            knt++;  /* Convert from 0-based loop count to iteration count */

            /* New BETA is at most 1, at least SAFMIN */
            xnorm = cblas_snrm2(n - 1, x, incx);
            beta = -copysignf(slapy2(*alpha, xnorm), *alpha);
        }

        *tau = (beta - *alpha) / beta;
        cblas_sscal(n - 1, ONE / (*alpha - beta), x, incx);

        /* If ALPHA is subnormal, it may lose relative accuracy */
        for (j = 0; j < knt; j++) {
            beta = beta * safmin;
        }
        *alpha = beta;
    }
}
