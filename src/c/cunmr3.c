/**
 * @file cunmr3.c
 * @brief CUNMR3 multiplies a general matrix by the unitary matrix from a RZ
 *        factorization determined by ctzrzf (unblocked algorithm).
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CUNMR3 overwrites the general complex m by n matrix C with
 *
 *       Q * C  if SIDE = 'L' and TRANS = 'N', or
 *
 *       Q**H* C  if SIDE = 'L' and TRANS = 'C', or
 *
 *       C * Q  if SIDE = 'R' and TRANS = 'N', or
 *
 *       C * Q**H if SIDE = 'R' and TRANS = 'C',
 *
 * where Q is a complex unitary matrix defined as the product of k
 * elementary reflectors
 *
 *       Q = H(1) H(2) . . . H(k)
 *
 * as returned by CTZRZF. Q is of order m if SIDE = 'L' and of order n
 * if SIDE = 'R'.
 *
 * @param[in]     side    = 'L': apply Q or Q**H from the Left
 *                         = 'R': apply Q or Q**H from the Right
 * @param[in]     trans   = 'N': apply Q  (No transpose)
 *                         = 'C': apply Q**H (Conjugate transpose)
 * @param[in]     m       The number of rows of the matrix C. m >= 0.
 * @param[in]     n       The number of columns of the matrix C. n >= 0.
 * @param[in]     k       The number of elementary reflectors whose product
 *                         defines the matrix Q.
 *                         If SIDE = 'L', m >= k >= 0;
 *                         if SIDE = 'R', n >= k >= 0.
 * @param[in]     l       The number of columns of the matrix A containing
 *                         the meaningful part of the Householder reflectors.
 *                         If SIDE = 'L', m >= l >= 0;
 *                         if SIDE = 'R', n >= l >= 0.
 * @param[in]     A       Complex*16 array, dimension
 *                                      (lda, m) if SIDE = 'L',
 *                                      (lda, n) if SIDE = 'R'
 *                         The i-th row must contain the vector which defines
 *                         the elementary reflector H(i), for i = 1,2,...,k,
 *                         as returned by CTZRZF in the last k rows of its
 *                         array argument A.
 *                         A is modified by the routine but restored on exit.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,k).
 * @param[in]     tau     Complex*16 array, dimension (k).
 *                         tau(i) must contain the scalar factor of the
 *                         elementary reflector H(i), as returned by CTZRZF.
 * @param[in,out] C       Complex*16 array, dimension (ldc, n).
 *                         On entry, the m-by-n matrix C.
 *                         On exit, C is overwritten by Q*C or Q**H*C or
 *                         C*Q**H or C*Q.
 * @param[in]     ldc     The leading dimension of the array C. ldc >= max(1,m).
 * @param[out]    work    Complex*16 array, dimension
 *                                      (n) if SIDE = 'L',
 *                                      (m) if SIDE = 'R'
 * @param[out]    info    = 0: successful exit
 *                         < 0: if info = -i, the i-th argument had an illegal
 *                              value
 */
void cunmr3(const char* side, const char* trans, const int m, const int n,
            const int k, const int l, c64* restrict A,
            const int lda, const c64* restrict tau,
            c64* restrict C, const int ldc,
            c64* restrict work, int* info)
{
    int left, notran;
    int i, i1, i2, i3, ic, ja, jc, mi = 0, ni = 0, nq;
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
    } else if (l < 0 || (left && (l > m)) ||
               (!left && (l > n))) {
        *info = -6;
    } else if (lda < (1 > k ? 1 : k)) {
        *info = -8;
    } else if (ldc < (1 > m ? 1 : m)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("CUNMR3", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) return;

    if ((left && !notran) || (!left && notran)) {
        i1 = 0;
        i2 = k;
        i3 = 1;
    } else {
        i1 = k - 1;
        i2 = -1;
        i3 = -1;
    }

    if (left) {
        ni = n;
        ja = m - l;
        jc = 0;
    } else {
        mi = m;
        ja = n - l;
        ic = 0;
    }

    for (i = i1; i != i2; i += i3) {
        if (left) {
            /* H(i) or H(i)**H is applied to C(i:m-1,0:n-1) */
            mi = m - i;
            ic = i;
        } else {
            /* H(i) or H(i)**H is applied to C(0:m-1,i:n-1) */
            ni = n - i;
            jc = i;
        }

        /* Apply H(i) or H(i)**H */
        if (notran) {
            taui = tau[i];
        } else {
            taui = conjf(tau[i]);
        }
        clarz(side, mi, ni, l, &A[i + ja * lda], lda, taui,
              &C[ic + jc * ldc], ldc, work);
    }
}
