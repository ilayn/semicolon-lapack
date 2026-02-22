/**
 * @file dlanst.c
 * @brief DLANST returns the value of the 1-norm, Frobenius norm, infinity
 *        norm, or the element of largest absolute value of a real symmetric
 *        tridiagonal matrix.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLANST returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real symmetric tridiagonal matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = '1', 'O' or 'o': norm1(A)
 *                  = 'I' or 'i': normI(A)
 *                  = 'F', 'f', 'E' or 'e': normF(A)
 * @param[in] n     The order of the matrix A. n >= 0.
 * @param[in] D     Double precision array, dimension (n).
 *                  The diagonal elements of A.
 * @param[in] E     Double precision array, dimension (n-1).
 *                  The (n-1) sub-diagonal or super-diagonal elements of A.
 *
 * @return The norm value.
 */
f64 dlanst(const char* norm, const INT n,
              const f64* restrict D, const f64* restrict E)
{
    f64 anorm;

    if (n <= 0) {
        return 0.0;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        anorm = fabs(D[n - 1]);
        for (INT i = 0; i < n - 1; i++) {
            f64 sum = fabs(D[i]);
            if (anorm < sum || isnan(sum)) anorm = sum;
            sum = fabs(E[i]);
            if (anorm < sum || isnan(sum)) anorm = sum;
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1' ||
               norm[0] == 'I' || norm[0] == 'i') {
        /* Find norm1(A) = normI(A) since A is symmetric */
        if (n == 1) {
            anorm = fabs(D[0]);
        } else {
            anorm = fabs(D[0]) + fabs(E[0]);
            f64 sum = fabs(E[n - 2]) + fabs(D[n - 1]);
            if (anorm < sum || isnan(sum)) anorm = sum;
            for (INT i = 1; i < n - 1; i++) {
                sum = fabs(D[i]) + fabs(E[i]) + fabs(E[i - 1]);
                if (anorm < sum || isnan(sum)) anorm = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) using dlassq */
        f64 scale = 0.0;
        f64 sum = 1.0;
        if (n > 1) {
            dlassq(n - 1, E, 1, &scale, &sum);
            sum = 2.0 * sum;
        }
        dlassq(n, D, 1, &scale, &sum);
        anorm = scale * sqrt(sum);
    } else {
        anorm = 0.0;
    }

    return anorm;
}
