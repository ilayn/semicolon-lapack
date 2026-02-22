/**
 * @file clanht.c
 * @brief CLANHT returns the value of the 1-norm, Frobenius norm, infinity
 *        norm, or the element of largest absolute value of a complex
 *        Hermitian tridiagonal matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLANHT returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex Hermitian tridiagonal matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = '1', 'O' or 'o': norm1(A)
 *                  = 'I' or 'i': normI(A)
 *                  = 'F', 'f', 'E' or 'e': normF(A)
 * @param[in] n     The order of the matrix A. n >= 0. When n = 0, CLANHT is
 *                  set to zero.
 * @param[in] D     Single precision array, dimension (n).
 *                  The diagonal elements of A.
 * @param[in] E     Complex*16 array, dimension (n-1).
 *                  The (n-1) sub-diagonal or super-diagonal elements of A.
 *
 * @return The norm value.
 */
f32 clanht(const char* norm, const INT n,
              const f32* restrict D,
              const c64* restrict E)
{
    f32 anorm;

    if (n <= 0) {
        return 0.0f;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))). */
        anorm = fabsf(D[n - 1]);
        for (INT i = 0; i < n - 1; i++) {
            f32 sum = fabsf(D[i]);
            if (anorm < sum || isnan(sum)) anorm = sum;
            sum = cabsf(E[i]);
            if (anorm < sum || isnan(sum)) anorm = sum;
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1' ||
               norm[0] == 'I' || norm[0] == 'i') {
        /* Find norm1(A). */
        if (n == 1) {
            anorm = fabsf(D[0]);
        } else {
            anorm = fabsf(D[0]) + cabsf(E[0]);
            f32 sum = cabsf(E[n - 2]) + fabsf(D[n - 1]);
            if (anorm < sum || isnan(sum)) anorm = sum;
            for (INT i = 1; i < n - 1; i++) {
                sum = fabsf(D[i]) + cabsf(E[i]) + cabsf(E[i - 1]);
                if (anorm < sum || isnan(sum)) anorm = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A). */
        f32 scale = 0.0f;
        f32 sum = 1.0f;
        if (n > 1) {
            classq(n - 1, E, 1, &scale, &sum);
            sum = 2.0f * sum;
        }
        slassq(n, D, 1, &scale, &sum);
        anorm = scale * sqrtf(sum);
    } else {
        anorm = 0.0f;
    }

    return anorm;
}
