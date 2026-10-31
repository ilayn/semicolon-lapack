/**
 * @file zupmtr.c
 * @brief ZUPMTR overwrites a general complex matrix with the product of Q
 *        from ZHPTRD and a matrix C, using packed storage.
 */

#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZUPMTR overwrites the general complex M-by-N matrix C with
 *
 *                 SIDE = 'L'     SIDE = 'R'
 * TRANS = 'N':      Q * C          C * Q
 * TRANS = 'C':      Q**H * C       C * Q**H
 *
 * where Q is a complex unitary matrix of order nq, with nq = m if
 * SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
 * nq-1 elementary reflectors, as returned by ZHPTRD using packed
 * storage:
 *
 * if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
 *
 * if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
 *
 * @param[in]     side   'L': apply Q or Q**H from the Left;
 *                       'R': apply Q or Q**H from the Right.
 * @param[in]     uplo   'U': Upper triangular packed storage used in previous
 *                             call to ZHPTRD;
 *                       'L': Lower triangular packed storage used in previous
 *                             call to ZHPTRD.
 * @param[in]     trans  'N': No transpose, apply Q;
 *                       'C': Conjugate transpose, apply Q**H.
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     AP     Complex array, dimension (m*(m+1)/2) if side = 'L'
 *                       or (n*(n+1)/2) if side = 'R'.
 *                       The vectors which define the elementary reflectors, as
 *                       returned by ZHPTRD. AP is modified by the routine but
 *                       restored on exit.
 * @param[in]     tau    Complex array, dimension (m-1) if side = 'L'
 *                       or (n-1) if side = 'R'.
 *                       tau(i) must contain the scalar factor of the elementary
 *                       reflector H(i), as returned by ZHPTRD.
 * @param[in,out] C      Complex array, dimension (ldc, n).
 *                       On entry, the M-by-N matrix C.
 *                       On exit, C is overwritten by Q*C or Q**H*C or C*Q**H
 *                       or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1,m).
 * @param[out]    work   Complex array, dimension (n) if side = 'L'
 *                       or (m) if side = 'R'.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void zupmtr(const char* side, const char* uplo, const char* trans,
            const INT m, const INT n,
            c128* restrict AP,
            const c128* restrict tau,
            c128* restrict C, const INT ldc,
            c128* restrict work,
            INT* info)
{
    INT left, notran, upper, forwrd;
    INT i, i1, i2, i3, ic, ii, jc, mi = 0, ni = 0, nq;
    c128 taui;

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
    } else if (!notran && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("ZUPMTR", -(*info));
        return;
    }

    /* Quick return if possible */

    if (m == 0 || n == 0) {
        return;
    }

    if (upper) {

        /* Q was determined by a call to ZHPTRD with UPLO = 'U' */

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

                /* H(i) or H(i)**H is applied to C(0:i,0:n-1) */

                mi = i + 1;
            } else {

                /* H(i) or H(i)**H is applied to C(0:m-1,0:i) */

                ni = i + 1;
            }

            /* Apply H(i) or H(i)**H */

            if (notran) {
                taui = tau[i];
            } else {
                taui = conj(tau[i]);
            }
            zlarf1l(side, mi, ni, &AP[ii - i], 1, taui, C,
                    ldc, work);

            if (forwrd) {
                ii = ii + i + 3;
            } else {
                ii = ii - i - 2;
            }
        }
    } else {

        /* Q was determined by a call to ZHPTRD with UPLO = 'L'. */

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
            if (left) {

                /* H(i) or H(i)**H is applied to C(i+1:m-1,0:n-1) */

                mi = m - i - 1;
                ic = i + 1;
            } else {

                /* H(i) or H(i)**H is applied to C(0:m-1,i+1:n-1) */

                ni = n - i - 1;
                jc = i + 1;
            }

            /* Apply H(i) or H(i)**H */

            if (notran) {
                taui = tau[i];
            } else {
                taui = conj(tau[i]);
            }
            zlarf1f(side, mi, ni, &AP[ii], 1, taui,
                    &C[ic + jc * ldc], ldc, work);

            if (forwrd) {
                ii = ii + nq - i;
            } else {
                ii = ii - nq + i - 1;
            }
        }
    }
}
