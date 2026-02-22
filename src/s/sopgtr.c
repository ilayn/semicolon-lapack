/**
 * @file sopgtr.c
 * @brief SOPGTR generates a real orthogonal matrix Q which is defined as the
 *        product of n-1 elementary reflectors H(i) of order n, as returned by
 *        SSPTRD using packed storage.
 */

#include "semicolon_lapack_single.h"

/**
 * SOPGTR generates a real orthogonal matrix Q which is defined as the
 * product of n-1 elementary reflectors H(i) of order n, as returned by
 * SSPTRD using packed storage:
 *
 * if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
 *
 * @param[in]     uplo   = 'U': Upper triangular packed storage used in previous
 *                              call to SSPTRD;
 *                       = 'L': Lower triangular packed storage used in previous
 *                              call to SSPTRD.
 * @param[in]     n      The order of the matrix Q. n >= 0.
 * @param[in]     AP     Double precision array, dimension (n*(n+1)/2).
 *                       The vectors which define the elementary reflectors.
 * @param[in]     tau    Double precision array, dimension (n-1).
 *                       tau[i] must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by SSPTRD.
 * @param[out]    Q      Double precision array, dimension (ldq, n).
 *                       The n-by-n orthogonal matrix Q.
 * @param[in]     ldq    The leading dimension of the array Q. ldq >= max(1, n).
 * @param[out]    work   Double precision array, dimension (n-1).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sopgtr(const char* uplo, const INT n, const f32* restrict AP,
            const f32* restrict tau, f32* restrict Q,
            const INT ldq, f32* restrict work, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

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
        xerbla("SOPGTR", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {

        /* Q was determined by a call to SSPTRD with UPLO = 'U'

           Unpack the vectors which define the elementary reflectors and
           set the last row and column of Q equal to those of the unit
           matrix */

        ij = 1;
        for (j = 0; j < n - 1; j++) {
            for (i = 0; i < j; i++) {
                Q[i + j * ldq] = AP[ij];
                ij = ij + 1;
            }
            ij = ij + 2;
            Q[(n - 1) + j * ldq] = ZERO;
        }
        for (i = 0; i < n - 1; i++) {
            Q[i + (n - 1) * ldq] = ZERO;
        }
        Q[(n - 1) + (n - 1) * ldq] = ONE;

        /* Generate Q(0:n-2,0:n-2) */

        sorg2l(n - 1, n - 1, n - 1, Q, ldq, tau, work, &iinfo);

    } else {

        /* Q was determined by a call to SSPTRD with UPLO = 'L'.

           Unpack the vectors which define the elementary reflectors and
           set the first row and column of Q equal to those of the unit
           matrix */

        Q[0 + 0 * ldq] = ONE;
        for (i = 1; i < n; i++) {
            Q[i + 0 * ldq] = ZERO;
        }
        ij = 2;
        for (j = 1; j < n; j++) {
            Q[0 + j * ldq] = ZERO;
            for (i = j + 1; i < n; i++) {
                Q[i + j * ldq] = AP[ij];
                ij = ij + 1;
            }
            ij = ij + 2;
        }
        if (n > 1) {

            /* Generate Q(1:n-1,1:n-1) */

            sorg2r(n - 1, n - 1, n - 1, &Q[1 + 1 * ldq], ldq, tau, work,
                   &iinfo);
        }
    }
}
