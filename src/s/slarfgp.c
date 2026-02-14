/**
 * @file slarfgp.c
 * @brief SLARFGP generates an elementary reflector (Householder matrix)
 *        with non-negative beta.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLARFGP generates a real elementary reflector H of order n, such that
 *
 *       H * ( alpha ) = ( beta ),   H**T * H = I.
 *           (   x   )   (   0  )
 *
 * where alpha and beta are scalars, beta is non-negative, and x is
 * an (n-1)-element real vector. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v**T ),
 *                     ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element vector.
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit matrix.
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
void slarfgp(const int n, f32 *alpha, f32 * const restrict x,
             const int incx, f32 *tau)
{
    const f32 TWO = 2.0f;
    const f32 ONE = 1.0f;
    f32 xnorm, beta, bignum, eps, savealpha, smlnum;
    int j, knt;

    if (n <= 0) {
        *tau = 0.0f;
        return;
    }

    eps = FLT_EPSILON;
    xnorm = cblas_snrm2(n - 1, x, incx);

    if (xnorm <= eps * fabsf(*alpha)) {
        /*
         * H = [+/-1, 0; I], sign chosen so ALPHA >= 0.
         */
        if (*alpha >= 0.0f) {
            /* When TAU == 0, the vector is special-cased to be
             * all zeros in the application routines. We do not need
             * to clear it. */
            *tau = 0.0f;
        } else {
            /* However, the application routines rely on explicit
             * zero checks when TAU != 0, and we must clear X. */
            *tau = TWO;
            for (j = 0; j < n - 1; j++) {
                x[j * incx] = 0.0f;
            }
            *alpha = -(*alpha);
        }
    } else {
        /*
         * General case
         */
        beta = copysignf(slapy2(*alpha, xnorm), *alpha);
        smlnum = FLT_MIN / FLT_EPSILON;
        knt = 0;

        if (fabsf(beta) < smlnum) {
            /*
             * XNORM, BETA may be inaccurate; scale X and recompute them
             */
            bignum = ONE / smlnum;
            while (fabsf(beta) < smlnum && knt < 20) {
                knt++;
                cblas_sscal(n - 1, bignum, x, incx);
                beta *= bignum;
                *alpha *= bignum;
            }

            /* New BETA is at most 1, at least SMLNUM */
            xnorm = cblas_snrm2(n - 1, x, incx);
            beta = copysignf(slapy2(*alpha, xnorm), *alpha);
        }

        savealpha = *alpha;
        *alpha += beta;

        if (beta < 0.0f) {
            beta = -beta;
            *tau = -(*alpha) / beta;
        } else {
            *alpha = xnorm * (xnorm / *alpha);
            *tau = *alpha / beta;
            *alpha = -(*alpha);
        }

        if (fabsf(*tau) <= smlnum) {
            /*
             * In the case where the computed TAU ends up being a
             * denormalized number, it loses relative accuracy. This is
             * a BIG problem. Solution: flush TAU to ZERO (or TWO).
             *
             * (Bug report provided by Pat Quillen from MathWorks
             *  on Jul 29, 2009.)
             */
            if (savealpha >= 0.0f) {
                *tau = 0.0f;
            } else {
                *tau = TWO;
                for (j = 0; j < n - 1; j++) {
                    x[j * incx] = 0.0f;
                }
                beta = -savealpha;
            }
        } else {
            /*
             * This is the general case.
             */
            cblas_sscal(n - 1, ONE / *alpha, x, incx);
        }

        /* If BETA is subnormal, it may lose relative accuracy */
        for (j = 0; j < knt; j++) {
            beta *= smlnum;
        }
        *alpha = beta;
    }
}
