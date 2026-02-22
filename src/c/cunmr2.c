/**
 * @file cunmr2.c
 * @brief CUNMR2 multiplies a general matrix by the unitary matrix from a
 *        RQ factorization determined by CGERQF (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CUNMR2 overwrites the general complex m-by-n matrix C with
 *
 *    Q * C   if SIDE = 'L' and TRANS = 'N', or
 *
 *    Q**H* C if SIDE = 'L' and TRANS = 'C', or
 *
 *    C * Q   if SIDE = 'R' and TRANS = 'N', or
 *
 *    C * Q**H if SIDE = 'R' and TRANS = 'C',
 *
 * where Q is a complex unitary matrix defined as the product of k
 * elementary reflectors
 *
 *    Q = H(1)**H H(2)**H . . . H(k)**H
 *
 * as returned by CGERQF. Q is of order m if SIDE = 'L' and of order n
 * if SIDE = 'R'.
 *
 * @param[in]     side   'L': apply Q or Q**H from the Left;
 *                       'R': apply Q or Q**H from the Right.
 * @param[in]     trans  'N': apply Q (No transpose);
 *                       'C': apply Q**H (Conjugate transpose).
 * @param[in]     m      The number of rows of C. m >= 0.
 * @param[in]     n      The number of columns of C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                       defines the matrix Q.
 *                       If SIDE = 'L', m >= k >= 0;
 *                       if SIDE = 'R', n >= k >= 0.
 * @param[in]     A      COMPLEX*16 array, dimension
 *                       (lda, m) if SIDE = 'L',
 *                       (lda, n) if SIDE = 'R'.
 *                       The i-th row must contain the vector which defines
 *                       the elementary reflector H(i), for i = 0,1,...,k-1,
 *                       as returned by CGERQF in the last k rows of its
 *                       array argument A.
 *                       A is modified by the routine but restored on exit.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, k).
 * @param[in]     tau    COMPLEX*16 array, dimension (k).
 *                       tau[i] must contain the scalar factor of the
 *                       elementary reflector H(i), as returned by CGERQF.
 * @param[in,out] C      COMPLEX*16 array, dimension (ldc, n).
 *                       On entry, the m-by-n matrix C.
 *                       On exit, C is overwritten by Q*C or Q**H*C or
 *                       C*Q**H or C*Q.
 * @param[in]     ldc    The leading dimension of C. ldc >= max(1, m).
 * @param[out]    work   COMPLEX*16 array, dimension
 *                       (n) if SIDE = 'L',
 *                       (m) if SIDE = 'R'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an
 *                                illegal value.
 */
void cunmr2(const char* side, const char* trans,
            const INT m, const INT n, const INT k,
            c64* restrict A, const INT lda,
            const c64* restrict tau,
            c64* restrict C, const INT ldc,
            c64* restrict work,
            INT* info)
{
    INT left, notran;
    INT i, i1, i2, i3, mi, ni, nq;
    c64 taui;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    /* NQ is the order of Q */
    if (left) {
        nq = m;
    } else {
        nq = n;
    }
    if (!left && !(side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (lda < (k > 1 ? k : 1)) {
        *info = -7;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("CUNMR2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    if ((left && !notran) || (!left && notran)) {
        i1 = 0;
        i2 = k - 1;
        i3 = 1;
    } else {
        i1 = k - 1;
        i2 = 0;
        i3 = -1;
    }

    if (left) {
        ni = n;
    } else {
        mi = m;
    }

    for (i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
        if (left) {
            /* H(i) or H(i)**H is applied to C(0:m-k+i, 0:n-1) */
            mi = m - k + i + 1;
        } else {
            /* H(i) or H(i)**H is applied to C(0:m-1, 0:n-k+i) */
            ni = n - k + i + 1;
        }

        /* Apply H(i) or H(i)**H */
        if (notran) {
            taui = conjf(tau[i]);
        } else {
            taui = tau[i];
        }
        clacgv(nq - k + i, &A[i + 0 * lda], lda);
        clarf1l(side, mi, ni, &A[i + 0 * lda], lda, taui, C, ldc, work);
        clacgv(nq - k + i, &A[i + 0 * lda], lda);
    }
}
