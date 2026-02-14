/**
 * @file stpmlqt.c
 * @brief STPMLQT applies a real orthogonal matrix Q obtained from a
 *        "triangular-pentagonal" real block reflector H to a general
 *        real matrix C, which consists of two blocks A and B.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STPMLQT applies a real orthogonal matrix Q obtained from a
 * "triangular-pentagonal" real block reflector H to a general
 * real matrix C, which consists of two blocks A and B.
 *
 * @param[in]     side    'L': apply Q or Q**T from the Left;
 *                        'R': apply Q or Q**T from the Right.
 * @param[in]     trans   'N': No transpose, apply Q;
 *                        'T': Transpose, apply Q**T.
 * @param[in]     m       The number of rows of the matrix B. m >= 0.
 * @param[in]     n       The number of columns of the matrix B. n >= 0.
 * @param[in]     k       The number of elementary reflectors whose product
 *                        defines the matrix Q.
 * @param[in]     l       The order of the trapezoidal part of V.
 *                        k >= l >= 0. See Further Details.
 * @param[in]     mb      The block size used for the storage of T. k >= mb >= 1.
 *                        This must be the same value of mb used to generate T
 *                        in STPLQT.
 * @param[in]     V       Double precision array, dimension (ldv,k).
 *                        The i-th row must contain the vector which defines
 *                        the elementary reflector H(i), as returned by STPLQT
 *                        in B.
 * @param[in]     ldv     The leading dimension of V. ldv >= k.
 * @param[in]     T       Double precision array, dimension (ldt,k).
 *                        The upper triangular factors of the block reflectors
 *                        as returned by STPLQT, stored as a mb-by-k matrix.
 * @param[in]     ldt     The leading dimension of T. ldt >= mb.
 * @param[in,out] A       Double precision array, dimension (lda,n) if side='L'
 *                        or (lda,k) if side='R'.
 *                        On entry, the k-by-n or m-by-k matrix A.
 *                        On exit, A is overwritten by the corresponding block
 *                        of Q*C or Q**T*C or C*Q or C*Q**T.
 * @param[in]     lda     The leading dimension of A.
 *                        If side = 'L', lda >= max(1,k);
 *                        if side = 'R', lda >= max(1,m).
 * @param[in,out] B       Double precision array, dimension (ldb,n).
 *                        On entry, the m-by-n matrix B.
 *                        On exit, B is overwritten by the corresponding block
 *                        of Q*C or Q**T*C or C*Q or C*Q**T.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,m).
 * @param[out]    work    Double precision array. Dimension is n*mb if side='L',
 *                        or m*mb if side='R'.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void stpmlqt(const char* side, const char* trans, const int m, const int n,
             const int k, const int l, const int mb,
             const f32* restrict V, const int ldv,
             const f32* restrict T, const int ldt,
             f32* restrict A, const int lda,
             f32* restrict B, const int ldb,
             f32* restrict work, int* info)
{
    int left, right, tran, notran;
    int i, ib, nb, lb, kf, ldaq;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    right = (side[0] == 'R' || side[0] == 'r');
    tran = (trans[0] == 'T' || trans[0] == 't');
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
        xerbla("STPMLQT", -(*info));
        return;
    }

    if (m == 0 || n == 0 || k == 0) return;

    if (left && notran) {
        for (i = 0; i < k; i += mb) {
            ib = ((k - i) < mb) ? (k - i) : mb;
            nb = ((m - l + i + ib) < m) ? (m - l + i + ib) : m;
            /* LAPACK has identical branches */
            lb = 0;
            stprfb("L", "T", "F", "R", nb, n, ib, lb,
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
            stprfb("R", "N", "F", "R", m, nb, ib, lb,
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
            stprfb("L", "N", "F", "R", nb, n, ib, lb,
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
            stprfb("R", "T", "F", "R", m, nb, ib, lb,
                   &V[i], ldv, &T[i * ldt], ldt,
                   &A[i * lda], lda, B, ldb, work, m);
        }
    }
}
