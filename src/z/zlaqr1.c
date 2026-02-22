/**
 * @file zlaqr1.c
 * @brief ZLAQR1 sets a scalar multiple of the first column of the product of
 *        2-by-2 or 3-by-3 matrix H and specified shifts.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>

/**
 * Given a 2-by-2 or 3-by-3 matrix H, ZLAQR1 sets v to a scalar multiple of
 * the first column of the product
 *
 *    K = (H - s1*I)*(H - s2*I)
 *
 * scaling to avoid overflows and most underflows.
 *
 * This is useful for starting f64 implicit shift bulges in the QR algorithm.
 *
 * @param[in] n       Order of the matrix H. n must be either 2 or 3.
 * @param[in] H       Complex array, dimension (ldh, n).
 *                    The 2-by-2 or 3-by-3 matrix H.
 * @param[in] ldh     Leading dimension of H. ldh >= n.
 * @param[in] s1      First shift.
 * @param[in] s2      Second shift.
 * @param[out] v      Complex array, dimension (n).
 *                    A scalar multiple of the first column of the matrix K.
 */
void zlaqr1(const INT n, const c128* H, const INT ldh,
            const c128 s1, const c128 s2,
            c128* v)
{
    const c128 zero = CMPLX(0.0, 0.0);
    const f64 rzero = 0.0;

    c128 h21s, h31s;
    f64 s;

    /* Quick return if possible */
    if (n != 2 && n != 3) {
        return;
    }

    if (n == 2) {
        s = cabs1(H[0] - s2) + cabs1(H[1]);
        if (s == rzero) {
            v[0] = zero;
            v[1] = zero;
        } else {
            h21s = H[1] / s;
            v[0] = h21s * H[ldh] + (H[0] - s1) *
                   ((H[0] - s2) / s);
            v[1] = h21s * (H[0] + H[1 + ldh] - s1 - s2);
        }
    } else {
        s = cabs1(H[0] - s2) + cabs1(H[1]) +
            cabs1(H[2]);
        if (s == rzero) {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        } else {
            h21s = H[1] / s;
            h31s = H[2] / s;
            v[0] = (H[0] - s1) * ((H[0] - s2) / s) +
                   H[ldh] * h21s + H[2 * ldh] * h31s;
            v[1] = h21s * (H[0] + H[1 + ldh] - s1 - s2) +
                   H[1 + 2 * ldh] * h31s;
            v[2] = h31s * (H[0] + H[2 + 2 * ldh] - s1 - s2) +
                   h21s * H[2 + ldh];
        }
    }
}
