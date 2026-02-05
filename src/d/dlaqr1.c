/**
 * @file dlaqr1.c
 * @brief DLAQR1 sets a scalar multiple of the first column of the product of
 *        2-by-2 or 3-by-3 matrix H and specified shifts.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * Given a 2-by-2 or 3-by-3 matrix H, DLAQR1 sets v to a scalar multiple of
 * the first column of the product
 *
 *    K = (H - (sr1 + i*si1)*I)*(H - (sr2 + i*si2)*I)
 *
 * scaling to avoid overflows and most underflows. It is assumed that either
 *
 *    1) sr1 = sr2 and si1 = -si2
 * or
 *    2) si1 = si2 = 0.
 *
 * This is useful for starting double implicit shift bulges in the QR algorithm.
 *
 * @param[in] n       Order of the matrix H. n must be either 2 or 3.
 * @param[in] H       Double precision array, dimension (ldh, n).
 *                    The 2-by-2 or 3-by-3 matrix H.
 * @param[in] ldh     Leading dimension of H. ldh >= n.
 * @param[in] sr1     Real part of first shift.
 * @param[in] si1     Imaginary part of first shift.
 * @param[in] sr2     Real part of second shift.
 * @param[in] si2     Imaginary part of second shift.
 * @param[out] v      Double precision array, dimension (n).
 *                    A scalar multiple of the first column of the matrix K.
 */
SEMICOLON_API void dlaqr1(const int n, const double* H, const int ldh,
                          const double sr1, const double si1,
                          const double sr2, const double si2,
                          double* v)
{
    const double zero = 0.0;

    double h21s, h31s, s;

    /* Quick return if possible */
    if (n != 2 && n != 3) {
        return;
    }

    if (n == 2) {
        /* 2-by-2 case */
        s = fabs(H[0] - sr2) + fabs(si2) + fabs(H[1]);
        if (s == zero) {
            v[0] = zero;
            v[1] = zero;
        } else {
            h21s = H[1] / s;
            v[0] = h21s * H[ldh] + (H[0] - sr1) * ((H[0] - sr2) / s) -
                   si1 * (si2 / s);
            v[1] = h21s * (H[0] + H[1 + ldh] - sr1 - sr2);
        }
    } else {
        /* 3-by-3 case */
        s = fabs(H[0] - sr2) + fabs(si2) + fabs(H[1]) + fabs(H[2]);
        if (s == zero) {
            v[0] = zero;
            v[1] = zero;
            v[2] = zero;
        } else {
            h21s = H[1] / s;
            h31s = H[2] / s;
            v[0] = (H[0] - sr1) * ((H[0] - sr2) / s) - si1 * (si2 / s) +
                   H[ldh] * h21s + H[2 * ldh] * h31s;
            v[1] = h21s * (H[0] + H[1 + ldh] - sr1 - sr2) +
                   H[1 + 2 * ldh] * h31s;
            v[2] = h31s * (H[0] + H[2 + 2 * ldh] - sr1 - sr2) +
                   h21s * H[2 + ldh];
        }
    }
}
