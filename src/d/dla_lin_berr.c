/**
 * @file dla_lin_berr.c
 * @brief DLA_LIN_BERR computes a component-wise relative backward error.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DLA_LIN_BERR computes componentwise relative backward error from
 * the formula
 *     max(i) ( abs(R(i)) / ( abs(op(A_s))*abs(Y) + abs(B_s) )(i) )
 * where abs(Z) is the componentwise absolute value of the matrix
 * or vector Z.
 *
 * @param[in]     n       The number of linear equations, i.e., the order of the
 *                        matrix A. n >= 0.
 * @param[in]     nz      We add (nz+1)*dlamch( 'Safe minimum' ) to R(i) in the
 *                        numerator to guard against spuriously zero residuals.
 *                        Default value is n.
 * @param[in]     nrhs    The number of right hand sides, i.e., the number of
 *                        columns of the matrices AYB, RES, and BERR. nrhs >= 0.
 * @param[in]     RES     Double precision array, dimension (n, nrhs).
 *                        The residual matrix, i.e., the matrix R in the relative
 *                        backward error formula above.
 * @param[in]     AYB     Double precision array, dimension (n, nrhs).
 *                        The denominator in the relative backward error formula
 *                        above, i.e., the matrix abs(op(A_s))*abs(Y) + abs(B_s).
 * @param[out]    BERR    Double precision array, dimension (nrhs).
 *                        The componentwise relative backward error from the
 *                        formula above.
 */
void dla_lin_berr(
    const INT n,
    const INT nz,
    const INT nrhs,
    const f64* restrict RES,
    const f64* restrict AYB,
    f64* restrict BERR)
{
    INT i, j;
    f64 tmp;
    f64 safe1;

    /*     Adding SAFE1 to the numerator guards against spuriously zero */
    /*     residuals.  A similar safeguard is in the DLA_yyAMV routine used */
    /*     to compute AYB. */

    safe1 = dlamch("Safe minimum");
    safe1 = (nz + 1) * safe1;

    for (j = 0; j < nrhs; j++) {
        BERR[j] = 0.0;
        for (i = 0; i < n; i++) {
            if (AYB[i + j * n] != 0.0) {
                tmp = (safe1 + fabs(RES[i + j * n])) / AYB[i + j * n];
                if (BERR[j] < tmp) {
                    BERR[j] = tmp;
                }
            }

            /*     If AYB is exactly 0.0 (and if computed by DLA_yyAMV), then we know */
            /*     the true residual also must be exactly 0.0. */

        }
    }
}
