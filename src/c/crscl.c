/**
 * @file crscl.c
 * @brief CRSCL multiplies a vector by the reciprocal of a complex scalar.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CRSCL multiplies an n-element complex vector x by the complex scalar
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
void crscl(
    const INT n,
    const c64 a,
    c64* restrict x,
    const INT incx)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    f32 safmin, safmax, ov, ar, ai, absr, absi, ur, ui;

    if (n <= 0) {
        return;
    }

    safmin = slamch("S");
    safmax = ONE / safmin;
    ov = slamch("O");

    ar = crealf(a);
    ai = cimagf(a);
    absr = fabsf(ar);
    absi = fabsf(ai);

    if (ai == ZERO) {
        cdrscl(n, ar, x, incx);

    } else if (ar == ZERO) {
        if (absi > safmax) {
            cblas_csscal(n, safmin, x, incx);
            c64 sc1 = CMPLXF(ZERO, -safmax / ai);
            cblas_cscal(n, &sc1, x, incx);
        } else if (absi < safmin) {
            c64 sc2 = CMPLXF(ZERO, -safmin / ai);
            cblas_cscal(n, &sc2, x, incx);
            cblas_csscal(n, safmax, x, incx);
        } else {
            c64 sc3 = CMPLXF(ZERO, -ONE / ai);
            cblas_cscal(n, &sc3, x, incx);
        }

    } else {
        ur = ar + ai * (ai / ar);
        ui = ai + ar * (ar / ai);

        if ((fabsf(ur) < safmin) || (fabsf(ui) < safmin)) {
            c64 sc4 = CMPLXF(safmin / ur, -safmin / ui);
            cblas_cscal(n, &sc4, x, incx);
            cblas_csscal(n, safmax, x, incx);
        } else if ((fabsf(ur) > safmax) || (fabsf(ui) > safmax)) {
            if ((absr > ov) || (absi > ov)) {
                c64 sc5 = CMPLXF(ONE / ur, -ONE / ui);
                cblas_cscal(n, &sc5, x, incx);
            } else {
                cblas_csscal(n, safmin, x, incx);
                if ((fabsf(ur) > ov) || (fabsf(ui) > ov)) {
                    if (absr >= absi) {
                        ur = (safmin * ar) + safmin * (ai * (ai / ar));
                        ui = (safmin * ai) + ar * ((safmin * ar) / ai);
                    } else {
                        ur = (safmin * ar) + ai * ((safmin * ai) / ar);
                        ui = (safmin * ai) + safmin * (ar * (ar / ai));
                    }
                    c64 sc6 = CMPLXF(ONE / ur, -ONE / ui);
                    cblas_cscal(n, &sc6, x, incx);
                } else {
                    c64 sc7 = CMPLXF(safmax / ur, -safmax / ui);
                    cblas_cscal(n, &sc7, x, incx);
                }
            }
        } else {
            c64 sc8 = CMPLXF(ONE / ur, -ONE / ui);
            cblas_cscal(n, &sc8, x, incx);
        }
    }
}
