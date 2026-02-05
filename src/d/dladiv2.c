/**
 * @file dladiv2.c
 * @brief DLADIV2 is a helper routine for DLADIV1, performing part of complex
 *        division using the Baudin-Smith algorithm.
 */

#include "semicolon_lapack_double.h"

/**
 * DLADIV2 is a helper routine for DLADIV1, performing part of complex
 * division using the Baudin-Smith algorithm.
 *
 * This function computes (A + B*R) * T or (A + D*(B/C)) * T depending
 * on the value of R. It is part of the robust complex division algorithm
 * described in "A Robust Complex Division in Scilab" by Baudin and Smith.
 *
 * @param[in] a  Double precision scalar.
 * @param[in] b  Double precision scalar.
 * @param[in] c  Double precision scalar.
 * @param[in] d  Double precision scalar.
 * @param[in] r  Double precision scalar, precomputed as D/C.
 * @param[in] t  Double precision scalar, precomputed as 1/(C + D*R).
 *
 * @return The computed value for use in complex division.
 */
double dladiv2(const double a, const double b, const double c,
               const double d, const double r, const double t)
{
    double br;

    if (r != 0.0) {
        br = b * r;
        if (br != 0.0) {
            return (a + br) * t;
        } else {
            return a * t + (b * t) * r;
        }
    } else {
        return (a + d * (b / c)) * t;
    }
}
