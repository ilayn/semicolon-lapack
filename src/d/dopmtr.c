/**
 * @file dopmtr.c
 * @brief DOPMTR overwrites the general real M-by-N matrix C with
 *        Q * C, Q**T * C, C * Q, or C * Q**T, where Q is defined as
 *        the product of elementary reflectors as returned by DSPTRD.
 */

#include "semicolon_lapack_double.h"

/**
 * DOPMTR overwrites the general real M-by-N matrix C with
 *
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'T':      Q**T * C       C * Q**T
 *
 * where Q is a real orthogonal matrix of order nq, with nq = m if
 * SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
 * nq-1 elementary reflectors, as returned by DSPTRD using packed
 * storage:
 *
 * if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
 *
 * @param[in]     side   = 'L': apply Q or Q**T from the Left;
 *                       = 'R': apply Q or Q**T from the Right.
 * @param[in]     uplo   = 'U': Upper triangular packed storage used in previous
 *                              call to DSPTRD;
 *                       = 'L': Lower triangular packed storage used in previous
 *                              call to DSPTRD.
 * @param[in]     trans  = 'N': No transpose, apply Q;
 *                       = 'T': Transpose, apply Q**T.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     AP     Double precision array, dimension
 *                       (m*(m+1)/2) if side = 'L'
 *                       (n*(n+1)/2) if side = 'R'
 *                       The vectors which define the elementary reflectors.
 *                       AP is modified by the routine but restored on exit.
 * @param[in]     tau    Double precision array, dimension (m-1) if side = 'L'
 *                       or (n-1) if side = 'R'.
 * @param[in,out] C      Double precision array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Double precision array, dimension (n) if side = 'L'
 *                       or (m) if side = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dopmtr(const char* side, const char* uplo, const char* trans,
            const int m, const int n, double* const restrict AP,
            const double* const restrict tau, double* const restrict C,
            const int ldc, double* const restrict work, int* info)
{
    const double ONE = 1.0;

    int forwrd, left, notran, upper;
    int i, i1, i2, i3, ic, ii, jc, mi, ni, nq;
    double aii;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    /* NQ is the order of Q */

    if (left) {
        nq = m;
    } else {
        nq = n;
    }
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DOPMTR", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    if (upper) {

        /* Q was determined by a call to DSPTRD with UPLO = 'U' */

        forwrd = (left && notran) || (!left && !notran);

        if (forwrd) {
            i1 = 0;
            i2 = nq - 2;
            i3 = 1;
            ii = 1;
        } else {
            i1 = nq - 2;
            i2 = 0;
            i3 = -1;
            ii = nq * (nq + 1) / 2 - 2;
        }

        if (left) {
            ni = n;
        } else {
            mi = m;
        }

        for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
            if (left) {

                /* H(i) is applied to C(0:i,0:n-1) */

                mi = i + 1;
            } else {

                /* H(i) is applied to C(0:m-1,0:i) */

                ni = i + 1;
            }

            /* Apply H(i) */

            dlarf1l(side, mi, ni, &AP[ii - i], 1, tau[i], C, ldc, work);

            if (forwrd) {
                ii = ii + i + 3;
            } else {
                ii = ii - i - 2;
            }
        }
    } else {

        /* Q was determined by a call to DSPTRD with UPLO = 'L'. */

        forwrd = (left && !notran) || (!left && notran);

        if (forwrd) {
            i1 = 0;
            i2 = nq - 2;
            i3 = 1;
            ii = 1;
        } else {
            i1 = nq - 2;
            i2 = 0;
            i3 = -1;
            ii = nq * (nq + 1) / 2 - 2;
        }

        if (left) {
            ni = n;
            jc = 0;
        } else {
            mi = m;
            ic = 0;
        }

        for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
            aii = AP[ii];
            AP[ii] = ONE;
            if (left) {

                /* H(i) is applied to C(i+1:m-1,0:n-1) */

                mi = m - i - 1;
                ic = i + 1;
            } else {

                /* H(i) is applied to C(0:m-1,i+1:n-1) */

                ni = n - i - 1;
                jc = i + 1;
            }

            /* Apply H(i) */

            dlarf(side, mi, ni, &AP[ii], 1, tau[i], &C[ic + jc * ldc], ldc, work);
            AP[ii] = aii;

            if (forwrd) {
                ii = ii + nq - i;
            } else {
                ii = ii - nq + i - 1;
            }
        }
    }
}
