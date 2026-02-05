/**
 * @file dlassq.c
 * @brief DLASSQ updates a sum of squares represented in scaled form.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/*
 * Blue's scaling constants for double precision.
 * These are computed from the IEEE 754 double precision parameters:
 *   radix = 2
 *   minexponent = -1021 (DBL_MIN_EXP - 1)
 *   maxexponent = 1024  (DBL_MAX_EXP)
 *   digits = 53         (DBL_MANT_DIG)
 *
 * tsml = radix^ceiling((minexponent - 1) * 0.5)
 *      = 2^ceiling((-1021 - 1) * 0.5) = 2^ceiling(-511) = 2^(-511)
 *
 * tbig = radix^floor((maxexponent - digits + 1) * 0.5)
 *      = 2^floor((1024 - 53 + 1) * 0.5) = 2^floor(486) = 2^486
 *
 * ssml = radix^(-floor((minexponent - digits) * 0.5))
 *      = 2^(-floor((-1021 - 53) * 0.5)) = 2^(-floor(-537)) = 2^537
 *
 * sbig = radix^(-ceiling((maxexponent + digits - 1) * 0.5))
 *      = 2^(-ceiling((1024 + 53 - 1) * 0.5)) = 2^(-ceiling(538)) = 2^(-538)
 */

/**
 * DLASSQ returns the values scale_out and sumsq_out such that
 *
 *    (scale_out**2)*sumsq_out = x(1)**2 +...+ x(n)**2 + (scale**2)*sumsq,
 *
 * where x(i) = X[incx * i] for 0 <= i < n (0-based indexing).
 * The value of sumsq is assumed to be non-negative.
 *
 * scale and sumsq must be supplied in SCALE and SUMSQ and
 * scale_out and sumsq_out are overwritten on SCALE and SUMSQ respectively.
 *
 * This routine uses the "Blue" algorithm for safe scaling, as described in:
 *   Anderson E. (2017) Algorithm 978: Safe Scaling in the Level 1 BLAS
 *   ACM Trans Math Softw 44:1--28
 *
 * @param[in]     n       The number of elements to be used from the vector x.
 * @param[in]     X       The vector for which a scaled sum of squares is computed.
 *                        Array of dimension (1 + (n-1)*abs(incx)).
 * @param[in]     incx    The increment between successive values of the vector x.
 *                        If incx > 0, X[incx * i] = x(i) for 0 <= i < n.
 *                        If incx < 0, X[incx * (n-1-i)] = x(i) for 0 <= i < n.
 *                        If incx = 0, x isn't a vector so there is no need to call
 *                        this subroutine. If you call it anyway, it will count x(0)
 *                        in the vector norm n times.
 * @param[in,out] scale   On entry, the value scale in the equation above.
 *                        On exit, SCALE is overwritten by scale_out, the scaling
 *                        factor for the sum of squares.
 * @param[in,out] sumsq   On entry, the value sumsq in the equation above.
 *                        On exit, SUMSQ is overwritten by sumsq_out, the basic sum
 *                        of squares from which scale_out has been factored out.
 */
void dlassq(
    const int n,
    const double * const restrict X,
    const int incx,
    double *scale,
    double *sumsq)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    /*
     * Blue's scaling constants for IEEE 754 double precision.
     * Using ldexp for exact powers of 2.
     */
    const double tsml = ldexp(1.0, -511);  /* 2^(-511) threshold for small */
    const double tbig = ldexp(1.0, 486);   /* 2^486 threshold for big */
    const double ssml = ldexp(1.0, 537);   /* 2^537 scale-up multiplier */
    const double sbig = ldexp(1.0, -538);  /* 2^(-538) scale-down multiplier */

    int i, ix;
    int notbig;
    double abig, amed, asml, ax, ymax, ymin;

    /* Quick return if possible */
    if (isnan(*scale) || isnan(*sumsq)) {
        return;
    }
    if (*sumsq == ZERO) {
        *scale = ONE;
    }
    if (*scale == ZERO) {
        *scale = ONE;
        *sumsq = ZERO;
    }
    if (n <= 0) {
        return;
    }

    /*
     * Compute the sum of squares in 3 accumulators:
     *    abig -- sums of squares scaled down to avoid overflow
     *    asml -- sums of squares scaled up to avoid underflow
     *    amed -- sums of squares that do not require scaling
     * The thresholds and multipliers are:
     *    tbig -- values bigger than this are scaled down by sbig
     *    tsml -- values smaller than this are scaled up by ssml
     */
    notbig = 1;
    asml = ZERO;
    amed = ZERO;
    abig = ZERO;

    /* Set starting index */
    if (incx < 0) {
        ix = -(n - 1) * incx;
    } else {
        ix = 0;
    }

    for (i = 0; i < n; i++) {
        ax = fabs(X[ix]);
        if (ax > tbig) {
            abig = abig + (ax * sbig) * (ax * sbig);
            notbig = 0;
        } else if (ax < tsml) {
            if (notbig) {
                asml = asml + (ax * ssml) * (ax * ssml);
            }
        } else {
            amed = amed + ax * ax;
        }
        ix = ix + incx;
    }

    /*
     * Put the existing sum of squares into one of the accumulators
     */
    if (*sumsq > ZERO) {
        ax = (*scale) * sqrt(*sumsq);
        if (ax > tbig) {
            if (*scale > ONE) {
                *scale = (*scale) * sbig;
                abig = abig + (*scale) * ((*scale) * (*sumsq));
            } else {
                /* sumsq > tbig^2 => (sbig * (sbig * sumsq)) is representable */
                abig = abig + (*scale) * ((*scale) * (sbig * (sbig * (*sumsq))));
            }
        } else if (ax < tsml) {
            if (notbig) {
                if (*scale < ONE) {
                    *scale = (*scale) * ssml;
                    asml = asml + (*scale) * ((*scale) * (*sumsq));
                } else {
                    /* sumsq < tsml^2 => (ssml * (ssml * sumsq)) is representable */
                    asml = asml + (*scale) * ((*scale) * (ssml * (ssml * (*sumsq))));
                }
            }
        } else {
            amed = amed + (*scale) * ((*scale) * (*sumsq));
        }
    }

    /*
     * Combine abig and amed or amed and asml if more than one
     * accumulator was used.
     */
    if (abig > ZERO) {
        /*
         * Combine abig and amed if abig > 0.
         */
        if (amed > ZERO || isnan(amed)) {
            abig = abig + (amed * sbig) * sbig;
        }
        *scale = ONE / sbig;
        *sumsq = abig;
    } else if (asml > ZERO) {
        /*
         * Combine amed and asml if asml > 0.
         */
        if (amed > ZERO || isnan(amed)) {
            amed = sqrt(amed);
            asml = sqrt(asml) / ssml;
            if (asml > amed) {
                ymin = amed;
                ymax = asml;
            } else {
                ymin = asml;
                ymax = amed;
            }
            *scale = ONE;
            *sumsq = ymax * ymax * (ONE + (ymin / ymax) * (ymin / ymax));
        } else {
            *scale = ONE / ssml;
            *sumsq = asml;
        }
    } else {
        /*
         * Otherwise all values are mid-range or zero
         */
        *scale = ONE;
        *sumsq = amed;
    }
}
