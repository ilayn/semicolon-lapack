/**
 * @file zunmtr.c
 * @brief ZUNMTR overwrites a general matrix with the product of the unitary
 *        matrix Q from ZHETRD.
 */

#include "semicolon_lapack_complex_double.h"
#include "../include/lapack_tuning.h"
#include <complex.h>

/**
 * ZUNMTR overwrites the general complex M-by-N matrix C with
 *
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'C':      Q**H * C       C * Q**H
 *
 * where Q is a complex unitary matrix of order nq, with nq = m if
 * SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
 * nq-1 elementary reflectors, as returned by ZHETRD:
 *
 * if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
 *
 * @param[in]     side   = 'L': apply Q or Q**H from the Left;
 *                         = 'R': apply Q or Q**H from the Right.
 * @param[in]     uplo   = 'U': Upper triangle of A contains elementary reflectors
 *                              from ZHETRD;
 *                         = 'L': Lower triangle of A contains elementary reflectors
 *                              from ZHETRD.
 * @param[in]     trans  = 'N': No transpose, apply Q;
 *                         = 'C': Conjugate transpose, apply Q**H.
 * @param[in]     m      The number of rows of C. m >= 0.
 * @param[in]     n      The number of columns of C. n >= 0.
 * @param[in]     A      Complex array, dimension (lda, m) if SIDE = 'L',
 *                       (lda, n) if SIDE = 'R'.
 *                       The vectors which define the elementary reflectors,
 *                       as returned by ZHETRD.
 * @param[in]     lda    The leading dimension of A.
 *                       lda >= max(1, m) if SIDE = 'L'; lda >= max(1, n) if SIDE = 'R'.
 * @param[in]     tau    Complex array, dimension (m-1) if SIDE = 'L',
 *                       (n-1) if SIDE = 'R'.
 *                       TAU(i) must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by ZHETRD.
 * @param[in,out] C      Complex array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   Complex array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If SIDE = 'L', lwork >= max(1, n);
 *                       if SIDE = 'R', lwork >= max(1, m).
 *                       For optimum performance lwork >= n*nb if SIDE = 'L', and
 *                       lwork >= m*nb if SIDE = 'R', where nb is the optimal blocksize.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zunmtr(const char* side, const char* uplo, const char* trans,
            const int m, const int n,
            const c128* const restrict A, const int lda,
            const c128* const restrict tau,
            c128* const restrict C, const int ldc,
            c128* const restrict work, const int lwork,
            int* info)
{
    int left, upper, lquery;
    int i1, i2, iinfo, lwkopt, mi, ni, nb, nq, nw;

    /* Test the input arguments */
    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n > 1 ? n : 1;
    } else {
        nq = n;
        nw = m > 1 ? m : 1;
    }

    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (nq > 1 ? nq : 1)) {
        *info = -7;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -10;
    } else if (lwork < nw && !lquery) {
        *info = -12;
    }

    if (*info == 0) {
        if (upper) {
            nb = lapack_get_nb("ORMQL");
        } else {
            nb = lapack_get_nb("ORMQR");
        }
        lwkopt = nw * nb;
        work[0] = CMPLX((f64)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZUNMTR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || nq == 1) {
        work[0] = CMPLX(1.0, 0.0);
        return;
    }

    if (left) {
        mi = m - 1;
        ni = n;
    } else {
        mi = m;
        ni = n - 1;
    }

    if (upper) {
        /* Q was determined by a call to ZHETRD with UPLO = 'U' */
        zunmql(side, trans, mi, ni, nq - 1,
               &A[0 + 1 * lda], lda, tau,
               C, ldc, work, lwork, &iinfo);
    } else {
        /* Q was determined by a call to ZHETRD with UPLO = 'L' */
        if (left) {
            i1 = 1;
            i2 = 0;
        } else {
            i1 = 0;
            i2 = 1;
        }
        zunmqr(side, trans, mi, ni, nq - 1,
               &A[1 + 0 * lda], lda, tau,
               &C[i1 + i2 * ldc], ldc, work, lwork, &iinfo);
    }

    work[0] = CMPLX((f64)lwkopt, 0.0);
}
