/**
 * @file zlanht.c
 * @brief ZLANHT returns the value of the 1-norm, Frobenius norm, infinity
 *        norm, or the element of largest absolute value of a complex
 *        Hermitian tridiagonal matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANHT returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex Hermitian tridiagonal matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = '1', 'O' or 'o': norm1(A)
 *                  = 'I' or 'i': normI(A)
 *                  = 'F', 'f', 'E' or 'e': normF(A)
 * @param[in] n     The order of the matrix A. n >= 0. When n = 0, ZLANHT is
 *                  set to zero.
 * @param[in] D     Double precision array, dimension (n).
 *                  The diagonal elements of A.
 * @param[in] E     Complex*16 array, dimension (n-1).
 *                  The (n-1) sub-diagonal or super-diagonal elements of A.
 *
 * @return The norm value.
 */
f64 zlanht(const char* norm, const int n,
              const f64* restrict D,
              const c128* restrict E)
{
    f64 anorm;

    if (n <= 0) {
        return 0.0;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))). */
        anorm = fabs(D[n - 1]);
        for (int i = 0; i < n - 1; i++) {
            f64 sum = fabs(D[i]);
            if (anorm < sum || isnan(sum)) anorm = sum;
            sum = cabs(E[i]);
            if (anorm < sum || isnan(sum)) anorm = sum;
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1' ||
               norm[0] == 'I' || norm[0] == 'i') {
        /* Find norm1(A). */
        if (n == 1) {
            anorm = fabs(D[0]);
        } else {
            anorm = fabs(D[0]) + cabs(E[0]);
            f64 sum = cabs(E[n - 2]) + fabs(D[n - 1]);
            if (anorm < sum || isnan(sum)) anorm = sum;
            for (int i = 1; i < n - 1; i++) {
                sum = fabs(D[i]) + cabs(E[i]) + cabs(E[i - 1]);
                if (anorm < sum || isnan(sum)) anorm = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A). */
        f64 scale = 0.0;
        f64 sum = 1.0;
        if (n > 1) {
            zlassq(n - 1, E, 1, &scale, &sum);
            sum = 2.0 * sum;
        }
        dlassq(n, D, 1, &scale, &sum);
        anorm = scale * sqrt(sum);
    } else {
        anorm = 0.0;
    }

    return anorm;
}
