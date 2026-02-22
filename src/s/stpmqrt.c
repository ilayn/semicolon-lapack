/**
 * @file stpmqrt.c
 * @brief STPMQRT applies a real orthogonal matrix Q obtained from a
 *        "triangular-pentagonal" real block reflector H to a general
 *        real matrix C, which consists of two blocks A and B.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STPMQRT applies a real orthogonal matrix Q obtained from a
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
 * @param[in]     nb      The block size used for the storage of T. k >= nb >= 1.
 *                        This must be the same value of nb used to generate T
 *                        in STPQRT.
 * @param[in]     V       Double precision array, dimension (ldv,k).
 *                        The i-th column must contain the vector which defines
 *                        the elementary reflector H(i), as returned by STPQRT
 *                        in B.
 * @param[in]     ldv     The leading dimension of V.
 *                        If side = 'L', ldv >= max(1,m);
 *                        if side = 'R', ldv >= max(1,n).
 * @param[in]     T       Double precision array, dimension (ldt,k).
 *                        The upper triangular factors of the block reflectors
 *                        as returned by STPQRT, stored as a nb-by-k matrix.
 * @param[in]     ldt     The leading dimension of T. ldt >= nb.
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
 * @param[out]    work    Double precision array. Dimension is n*nb if side='L',
 *                        or m*nb if side='R'.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void stpmqrt(const char* side, const char* trans, const INT m, const INT n,
             const INT k, const INT l, const INT nb,
             const f32* restrict V, const INT ldv,
             const f32* restrict T, const INT ldt,
             f32* restrict A, const INT lda,
             f32* restrict B, const INT ldb,
             f32* restrict work, INT* info)
{
    INT left, right, tran, notran;
    INT i, ib, mb, lb, kf, ldaq, ldvq;

    *info = 0;
    left = (side[0] == 'L' || side[0] == 'l');
    right = (side[0] == 'R' || side[0] == 'r');
    tran = (trans[0] == 'T' || trans[0] == 't');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (left) {
        ldvq = (m > 1) ? m : 1;
        ldaq = (k > 1) ? k : 1;
    } else if (right) {
        ldvq = (n > 1) ? n : 1;
        ldaq = (m > 1) ? m : 1;
    } else {
        ldvq = 1;
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
    } else if (nb < 1 || (nb > k && k > 0)) {
        *info = -7;
    } else if (ldv < ldvq) {
        *info = -9;
    } else if (ldt < nb) {
        *info = -11;
    } else if (lda < ldaq) {
        *info = -13;
    } else if (ldb < ((m > 1) ? m : 1)) {
        *info = -15;
    }

    if (*info != 0) {
        xerbla("STPMQRT", -(*info));
        return;
    }

    if (m == 0 || n == 0 || k == 0) return;

    if (left && tran) {
        for (i = 0; i < k; i += nb) {
            ib = ((k - i) < nb) ? (k - i) : nb;
            mb = ((m - l + i + ib) < m) ? (m - l + i + ib) : m;
            if (i >= l) {
                lb = 0;
            } else {
                lb = mb - m + l - i;
            }
            stprfb("L", "T", "F", "C", mb, n, ib, lb,
                   &V[i * ldv], ldv, &T[i * ldt], ldt,
                   &A[i], lda, B, ldb, work, ib);
        }

    } else if (right && notran) {
        for (i = 0; i < k; i += nb) {
            ib = ((k - i) < nb) ? (k - i) : nb;
            mb = ((n - l + i + ib) < n) ? (n - l + i + ib) : n;
            if (i >= l) {
                lb = 0;
            } else {
                lb = mb - n + l - i;
            }
            stprfb("R", "N", "F", "C", m, mb, ib, lb,
                   &V[i * ldv], ldv, &T[i * ldt], ldt,
                   &A[i * lda], lda, B, ldb, work, m);
        }

    } else if (left && notran) {
        kf = ((k - 1) / nb) * nb;
        for (i = kf; i >= 0; i -= nb) {
            ib = ((k - i) < nb) ? (k - i) : nb;
            mb = ((m - l + i + ib) < m) ? (m - l + i + ib) : m;
            if (i >= l) {
                lb = 0;
            } else {
                lb = mb - m + l - i;
            }
            stprfb("L", "N", "F", "C", mb, n, ib, lb,
                   &V[i * ldv], ldv, &T[i * ldt], ldt,
                   &A[i], lda, B, ldb, work, ib);
        }

    } else if (right && tran) {
        kf = ((k - 1) / nb) * nb;
        for (i = kf; i >= 0; i -= nb) {
            ib = ((k - i) < nb) ? (k - i) : nb;
            mb = ((n - l + i + ib) < n) ? (n - l + i + ib) : n;
            if (i >= l) {
                lb = 0;
            } else {
                lb = mb - n + l - i;
            }
            stprfb("R", "T", "F", "C", m, mb, ib, lb,
                   &V[i * ldv], ldv, &T[i * ldt], ldt,
                   &A[i * lda], lda, B, ldb, work, m);
        }
    }
}
