/**
 * @file zunm2r.c
 * @brief ZUNM2R multiplies a general matrix by the unitary matrix from a QR
 *        factorization determined by ZGEQRF (unblocked algorithm).
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZUNM2R overwrites the general complex m-by-n matrix C with
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
 * as returned by ZGEQRF. Q is of order m if SIDE = 'L' and of order n
 * if SIDE = 'R'.
 *
 * @param[in]     side   = 'L': apply Q or Q**H from the Left
 *                         = 'R': apply Q or Q**H from the Right
 * @param[in]     trans  = 'N': apply Q  (No transpose)
 *                         = 'C': apply Q**H (Conjugate transpose)
 * @param[in]     m      The number of rows of the matrix C. m >= 0.
 * @param[in]     n      The number of columns of the matrix C. n >= 0.
 * @param[in]     k      The number of elementary reflectors whose product
 *                        defines the matrix Q.
 *                        If SIDE = 'L', m >= k >= 0;
 *                        if SIDE = 'R', n >= k >= 0.
 * @param[in]     A      Complex array, dimension (lda, k).
 *                        The i-th column must contain the vector which defines
 *                        the elementary reflector H(i), for i = 0,1,...,k-1,
 *                        as returned by ZGEQRF in the first k columns of its
 *                        array argument A.
 *                        A is modified by the routine but restored on exit.
 * @param[in]     lda    The leading dimension of the array A.
 *                        If SIDE = 'L', lda >= max(1, m);
 *                        if SIDE = 'R', lda >= max(1, n).
 * @param[in]     tau    Complex array, dimension (k).
 *                        tau(i) must contain the scalar factor of the
 *                        elementary reflector H(i), as returned by ZGEQRF.
 * @param[in,out] C      Complex array, dimension (ldc, n).
 *                        On entry, the m-by-n matrix C.
 *                        On exit, C is overwritten by Q*C or Q**H*C or
 *                        C*Q**H or C*Q.
 * @param[in]     ldc    The leading dimension of the array C. ldc >= max(1, m).
 * @param[out]    work   Complex workspace array, dimension
 *                        (n) if SIDE = 'L',
 *                        (m) if SIDE = 'R'.
 * @param[out]    info   = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal
 *                             value.
 */
void zunm2r(const char* side, const char* trans,
            const INT m, const INT n, const INT k,
            const c128* restrict A, const INT lda,
            const c128* restrict tau,
            c128* restrict C, const INT ldc,
            c128* restrict work,
            INT* info)
{
    INT left, notran;
    INT i, i1, i2, i3, ic, jc, mi = 0, ni = 0, nq;
    c128 taui;

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
    } else if (lda < (nq > 1 ? nq : 1)) {
        *info = -7;
    } else if (ldc < (m > 1 ? m : 1)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("ZUNM2R", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

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
        jc = 0;
    } else {
        mi = m;
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
            taui = conj(tau[i]);
        }
        zlarf1f(side, mi, ni, &A[i + i * lda], 1, taui,
                &C[ic + jc * ldc], ldc, work);
    }
}
