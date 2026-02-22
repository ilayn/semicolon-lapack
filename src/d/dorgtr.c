/**
 * @file dorgtr.c
 * @brief DORGTR generates a real orthogonal matrix Q which is defined as the
 *        product of n-1 elementary reflectors of order N, as returned by DSYTRD.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include "../include/lapack_tuning.h"

/**
 * DORGTR generates a real orthogonal matrix Q which is defined as the
 * product of n-1 elementary reflectors of order N, as returned by
 * DSYTRD:
 *
 * if UPLO = 'U', Q = H(n-1) . . . H(2) H(1),
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(n-1).
 *
 * @param[in]     uplo   = 'U': Upper triangle of A contains elementary reflectors
 *                              from DSYTRD;
 *                         = 'L': Lower triangle of A contains elementary reflectors
 *                              from DSYTRD.
 * @param[in]     n      The order of the matrix Q. n >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the vectors which define the elementary reflectors,
 *                       as returned by DSYTRD.
 *                       On exit, the N-by-N orthogonal matrix Q.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     tau    Double precision array, dimension (n-1).
 *                       TAU(i) must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by DSYTRD.
 * @param[out]    work   Double precision array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work. lwork >= max(1, n-1).
 *                       For optimum performance lwork >= (n-1)*nb, where nb is
 *                       the optimal blocksize.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dorgtr(const char* uplo, const INT n,
            f64* restrict A, const INT lda,
            const f64* restrict tau,
            f64* restrict work, const INT lwork,
            INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT upper, lquery;
    INT i, j, iinfo, lwkopt, nb;

    /* Test the input arguments */
    *info = 0;
    lquery = (lwork == -1);
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (lwork < (n - 1 > 1 ? n - 1 : 1) && !lquery) {
        *info = -7;
    }

    if (*info == 0) {
        if (upper) {
            nb = lapack_get_nb("ORGQL");
        } else {
            nb = lapack_get_nb("ORGQR");
        }
        lwkopt = (n - 1 > 1 ? n - 1 : 1) * nb;
        work[0] = (f64)lwkopt;
    }

    if (*info != 0) {
        xerbla("DORGTR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        work[0] = 1.0;
        return;
    }

    if (upper) {
        /* Q was determined by a call to DSYTRD with UPLO = 'U'
         *
         * Shift the vectors which define the elementary reflectors one
         * column to the left, and set the last row and column of Q to
         * those of the unit matrix */

        for (j = 0; j < n - 1; j++) {
            for (i = 0; i < j; i++) {
                A[i + j * lda] = A[i + (j + 1) * lda];
            }
            A[n - 1 + j * lda] = ZERO;
        }
        for (i = 0; i < n - 1; i++) {
            A[i + (n - 1) * lda] = ZERO;
        }
        A[(n - 1) + (n - 1) * lda] = ONE;

        /* Generate Q(0:n-2, 0:n-2) */
        if (n - 1 > 0) {
            dorgql(n - 1, n - 1, n - 1, A, lda, tau, work, lwork, &iinfo);
        }

    } else {
        /* Q was determined by a call to DSYTRD with UPLO = 'L'.
         *
         * Shift the vectors which define the elementary reflectors one
         * column to the right, and set the first row and column of Q to
         * those of the unit matrix */

        for (j = n - 1; j >= 1; j--) {
            A[0 + j * lda] = ZERO;
            for (i = j + 1; i < n; i++) {
                A[i + j * lda] = A[i + (j - 1) * lda];
            }
        }
        A[0 + 0 * lda] = ONE;
        for (i = 1; i < n; i++) {
            A[i + 0 * lda] = ZERO;
        }

        if (n > 1) {
            /* Generate Q(1:n-1, 1:n-1) */
            dorgqr(n - 1, n - 1, n - 1, &A[1 + 1 * lda], lda, tau,
                   work, lwork, &iinfo);
        }
    }

    work[0] = (f64)lwkopt;
}
