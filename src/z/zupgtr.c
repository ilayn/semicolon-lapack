/**
 * @file zupgtr.c
 * @brief ZUPGTR generates a complex unitary matrix Q.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZUPGTR generates a complex unitary matrix Q which is defined as the
 * product of n-1 elementary reflectors H(i) of order n, as returned by
 * ZHPTRD using packed storage:
 *
 * if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
 *
 * @param[in]     uplo   = 'U': Upper triangular packed storage used in previous
 *                               call to ZHPTRD;
 *                         = 'L': Lower triangular packed storage used in previous
 *                               call to ZHPTRD.
 * @param[in]     n      The order of the matrix Q. N >= 0.
 * @param[in]     AP     Double complex array, dimension (N*(N+1)/2).
 *                       The vectors which define the elementary reflectors, as
 *                       returned by ZHPTRD.
 * @param[in]     tau    Double complex array, dimension (N-1).
 *                       TAU(i) must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by ZHPTRD.
 * @param[out]    Q      Double complex array, dimension (LDQ,N).
 *                       The N-by-N unitary matrix Q.
 * @param[in]     ldq    The leading dimension of the array Q. LDQ >= max(1,N).
 * @param[out]    work   Double complex array, dimension (N-1).
 * @param[out]    info   = 0:  successful exit
 *                       < 0:  if INFO = -i, the i-th argument had an illegal value
 */
void zupgtr(const char* uplo, const INT n, const c128* AP,
            const c128* tau, c128* Q, const INT ldq,
            c128* work, INT* info) {

    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

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
        xerbla("ZUPGTR", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) {
        return;
    }

    if (upper) {

        // Q was determined by a call to ZHPTRD with UPLO = 'U'
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
        zung2l(n - 1, n - 1, n - 1, Q, ldq, tau, work, &iinfo);

    } else {

        // Q was determined by a call to ZHPTRD with UPLO = 'L'.
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
            zung2r(n - 1, n - 1, n - 1, &Q[1 + ldq], ldq, tau, work,
                   &iinfo);
        }
    }
}
