/**
 * @file ztpmlqt.c
 * @brief ZTPMLQT applies a complex unitary matrix Q obtained from a
 *        "triangular-pentagonal" complex block reflector H to a general
 *        complex matrix C, which consists of two blocks A and B.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTPMLQT applies a complex unitary matrix Q obtained from a
 * "triangular-pentagonal" complex block reflector H to a general
 * complex matrix C, which consists of two blocks A and B.
 *
 * @param[in]     side    'L': apply Q or Q**H from the Left;
 *                        'R': apply Q or Q**H from the Right.
 * @param[in]     trans   'N': No transpose, apply Q;
 *                        'C': Conjugate transpose, apply Q**H.
 * @param[in]     m       The number of rows of the matrix B. m >= 0.
 * @param[in]     n       The number of columns of the matrix B. n >= 0.
 * @param[in]     k       The number of elementary reflectors whose product
 *                        defines the matrix Q.
 * @param[in]     l       The order of the trapezoidal part of V.
 *                        k >= l >= 0. See Further Details.
 * @param[in]     mb      The block size used for the storage of T. k >= mb >= 1.
 *                        This must be the same value of mb used to generate T
 *                        in ZTPLQT.
 * @param[in]     V       Complex*16 array, dimension (ldv,k).
 *                        The i-th row must contain the vector which defines
 *                        the elementary reflector H(i), as returned by ZTPLQT
 *                        in B.
 * @param[in]     ldv     The leading dimension of V. ldv >= k.
 * @param[in]     T       Complex*16 array, dimension (ldt,k).
 *                        The upper triangular factors of the block reflectors
 *                        as returned by ZTPLQT, stored as a mb-by-k matrix.
 * @param[in]     ldt     The leading dimension of T. ldt >= mb.
 * @param[in,out] A       Complex*16 array, dimension (lda,n) if side='L'
 *                        or (lda,k) if side='R'.
 *                        On entry, the k-by-n or m-by-k matrix A.
 *                        On exit, A is overwritten by the corresponding block
 *                        of Q*C or Q**H*C or C*Q or C*Q**H.
 * @param[in]     lda     The leading dimension of A.
 *                        If side = 'L', lda >= max(1,k);
 *                        if side = 'R', lda >= max(1,m).
 * @param[in,out] B       Complex*16 array, dimension (ldb,n).
 *                        On entry, the m-by-n matrix B.
 *                        On exit, B is overwritten by the corresponding block
 *                        of Q*C or Q**H*C or C*Q or C*Q**H.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,m).
 * @param[out]    work    Complex*16 array. Dimension is n*mb if side='L',
 *                        or m*mb if side='R'.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ztpmlqt(const char* side, const char* trans, const int m, const int n,
             const int k, const int l, const int mb,
             const c128* const restrict V, const int ldv,
             const c128* const restrict T, const int ldt,
             c128* const restrict A, const int lda,
             c128* const restrict B, const int ldb,
             c128* const restrict work, int* info)
{
    int left, right, tran, notran;
    int i, ib, nb, lb, kf, ldaq;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    right = (side[0] == 'R' || side[0] == 'r');
    tran = (trans[0] == 'C' || trans[0] == 'c');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (left) {
        ldaq = (k > 1) ? k : 1;
    } else if (right) {
        ldaq = (m > 1) ? m : 1;
    } else {
        ldaq = 1;
    }

    if (!left && !right) {
        *info = -1;
    } else if (!tran && !notran) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0) {
        *info = -5;
    } else if (l < 0 || l > k) {
        *info = -6;
    } else if (mb < 1 || (mb > k && k > 0)) {
        *info = -7;
    } else if (ldv < k) {
        *info = -9;
    } else if (ldt < mb) {
        *info = -11;
    } else if (lda < ldaq) {
        *info = -13;
    } else if (ldb < ((m > 1) ? m : 1)) {
        *info = -15;
    }

    if (*info != 0) {
        xerbla("ZTPMLQT", -(*info));
        return;
    }

    if (m == 0 || n == 0 || k == 0) return;

    if (left && notran) {
        for (i = 0; i < k; i += mb) {
            ib = ((k - i) < mb) ? (k - i) : mb;
            nb = ((m - l + i + ib) < m) ? (m - l + i + ib) : m;
            /* LAPACK has identical branches */
            lb = 0;
            ztprfb("L", "C", "F", "R", nb, n, ib, lb,
                   &V[i], ldv, &T[i * ldt], ldt,
                   &A[i], lda, B, ldb, work, ib);
        }

    } else if (right && tran) {
        for (i = 0; i < k; i += mb) {
            ib = ((k - i) < mb) ? (k - i) : mb;
            nb = ((n - l + i + ib) < n) ? (n - l + i + ib) : n;
            if (i >= l) {
                lb = 0;
            } else {
                lb = nb - n + l - i;
            }
            ztprfb("R", "N", "F", "R", m, nb, ib, lb,
                   &V[i], ldv, &T[i * ldt], ldt,
                   &A[i * lda], lda, B, ldb, work, m);
        }

    } else if (left && tran) {
        kf = ((k - 1) / mb) * mb;
        for (i = kf; i >= 0; i -= mb) {
            ib = ((k - i) < mb) ? (k - i) : mb;
            nb = ((m - l + i + ib) < m) ? (m - l + i + ib) : m;
            /* LAPACK has identical branches */
            lb = 0;
            ztprfb("L", "N", "F", "R", nb, n, ib, lb,
                   &V[i], ldv, &T[i * ldt], ldt,
                   &A[i], lda, B, ldb, work, ib);
        }

    } else if (right && notran) {
        kf = ((k - 1) / mb) * mb;
        for (i = kf; i >= 0; i -= mb) {
            ib = ((k - i) < mb) ? (k - i) : mb;
            nb = ((n - l + i + ib) < n) ? (n - l + i + ib) : n;
            if (i >= l) {
                lb = 0;
            } else {
                lb = nb - n + l - i;
            }
            ztprfb("R", "C", "F", "R", m, nb, ib, lb,
                   &V[i], ldv, &T[i * ldt], ldt,
                   &A[i * lda], lda, B, ldb, work, m);
        }
    }
}
