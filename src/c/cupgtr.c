/**
 * @file cupgtr.c
 * @brief CUPGTR generates a complex unitary matrix Q.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CUPGTR generates a complex unitary matrix Q which is defined as the
 * product of n-1 elementary reflectors H(i) of order n, as returned by
 * CHPTRD using packed storage:
 *
 * if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
 *
 * @param[in]     uplo   = 'U': Upper triangular packed storage used in previous
 *                               call to CHPTRD;
 *                         = 'L': Lower triangular packed storage used in previous
 *                               call to CHPTRD.
 * @param[in]     n      The order of the matrix Q. N >= 0.
 * @param[in]     AP     Single complex array, dimension (N*(N+1)/2).
 *                       The vectors which define the elementary reflectors, as
 *                       returned by CHPTRD.
 * @param[in]     tau    Single complex array, dimension (N-1).
 *                       TAU(i) must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by CHPTRD.
 * @param[out]    Q      Single complex array, dimension (LDQ,N).
 *                       The N-by-N unitary matrix Q.
 * @param[in]     ldq    The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[out]    work   Single complex array, dimension (N-1).
 * @param[out]    info   = 0:  successful exit
 *                       < 0:  if INFO = -i, the i-th argument had an illegal value
 */
void cupgtr(const char* uplo, const INT n, const c64* AP,
            const c64* tau, c64* Q, const INT ldq,
            c64* work, INT* info) {

    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT upper;
    INT i, iinfo, ij, j;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldq < (1 > n ? 1 : n)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("CUPGTR", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) {
        return;
    }

    if (upper) {

        // Q was determined by a call to CHPTRD with UPLO = 'U'
        //
        // Unpack the vectors which define the elementary reflectors and
        // set the last row and column of Q equal to those of the unit
        // matrix

        ij = 1;  // Fortran IJ=2, but 0-based: points to AP[1] (second element)
        for (j = 0; j < n - 1; j++) {
            for (i = 0; i < j; i++) {
                Q[i + j * ldq] = AP[ij];
                ij++;
            }
            ij += 2;
            Q[(n - 1) + j * ldq] = CZERO;
        }
        for (i = 0; i < n - 1; i++) {
            Q[i + (n - 1) * ldq] = CZERO;
        }
        Q[(n - 1) + (n - 1) * ldq] = CONE;

        // Generate Q(0:n-2,0:n-2)
        cung2l(n - 1, n - 1, n - 1, Q, ldq, tau, work, &iinfo);

    } else {

        // Q was determined by a call to CHPTRD with UPLO = 'L'.
        //
        // Unpack the vectors which define the elementary reflectors and
        // set the first row and column of Q equal to those of the unit
        // matrix

        Q[0] = CONE;
        for (i = 1; i < n; i++) {
            Q[i] = CZERO;
        }
        ij = 2;  // Fortran IJ=3, 0-based: AP[2]
        for (j = 1; j < n; j++) {
            Q[j * ldq] = CZERO;
            for (i = j + 1; i < n; i++) {
                Q[i + j * ldq] = AP[ij];
                ij++;
            }
            ij += 2;
        }
        if (n > 1) {

            // Generate Q(1:n-1,1:n-1)
            cung2r(n - 1, n - 1, n - 1, &Q[1 + ldq], ldq, tau, work,
                   &iinfo);
        }
    }
}
