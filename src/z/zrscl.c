/**
 * @file zrscl.c
 * @brief ZRSCL multiplies a vector by the reciprocal of a complex scalar.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZRSCL multiplies an n-element complex vector x by the complex scalar
 * 1/a.  This is done without overflow or underflow as long as
 * the final result x/a does not overflow or underflow.
 *
 * @param[in]     n     The number of components of the vector x.
 * @param[in]     a     The scalar a which is used to divide each component of x.
 *                      a must not be 0, or the subroutine will divide by zero.
 * @param[in,out] x     Complex*16 array, dimension (1+(n-1)*abs(incx)).
 *                      The n-element vector x.
 * @param[in]     incx  The increment between successive values of the vector x.
 *                      incx > 0.
 */
void zrscl(
    const int n,
    const c128 a,
    c128* restrict x,
    const int incx)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    f64 safmin, safmax, ov, ar, ai, absr, absi, ur, ui;

    if (n <= 0) {
        return;
    }

    safmin = dlamch("S");
    safmax = ONE / safmin;
    ov = dlamch("O");

    ar = creal(a);
    ai = cimag(a);
    absr = fabs(ar);
    absi = fabs(ai);

    if (ai == ZERO) {
        zdrscl(n, ar, x, incx);

    } else if (ar == ZERO) {
        if (absi > safmax) {
            cblas_zdscal(n, safmin, x, incx);
            c128 sc1 = CMPLX(ZERO, -safmax / ai);
            cblas_zscal(n, &sc1, x, incx);
        } else if (absi < safmin) {
            c128 sc2 = CMPLX(ZERO, -safmin / ai);
            cblas_zscal(n, &sc2, x, incx);
            cblas_zdscal(n, safmax, x, incx);
        } else {
            c128 sc3 = CMPLX(ZERO, -ONE / ai);
            cblas_zscal(n, &sc3, x, incx);
        }

    } else {
        ur = ar + ai * (ai / ar);
        ui = ai + ar * (ar / ai);

        if ((fabs(ur) < safmin) || (fabs(ui) < safmin)) {
            c128 sc4 = CMPLX(safmin / ur, -safmin / ui);
            cblas_zscal(n, &sc4, x, incx);
            cblas_zdscal(n, safmax, x, incx);
        } else if ((fabs(ur) > safmax) || (fabs(ui) > safmax)) {
            if ((absr > ov) || (absi > ov)) {
                c128 sc5 = CMPLX(ONE / ur, -ONE / ui);
                cblas_zscal(n, &sc5, x, incx);
            } else {
                cblas_zdscal(n, safmin, x, incx);
                if ((fabs(ur) > ov) || (fabs(ui) > ov)) {
                    if (absr >= absi) {
                        ur = (safmin * ar) + safmin * (ai * (ai / ar));
                        ui = (safmin * ai) + ar * ((safmin * ar) / ai);
                    } else {
                        ur = (safmin * ar) + ai * ((safmin * ai) / ar);
                        ui = (safmin * ai) + safmin * (ar * (ar / ai));
                    }
                    c128 sc6 = CMPLX(ONE / ur, -ONE / ui);
                    cblas_zscal(n, &sc6, x, incx);
                } else {
                    c128 sc7 = CMPLX(safmax / ur, -safmax / ui);
                    cblas_zscal(n, &sc7, x, incx);
                }
            }
        } else {
            c128 sc8 = CMPLX(ONE / ur, -ONE / ui);
            cblas_zscal(n, &sc8, x, incx);
        }
    }
}
