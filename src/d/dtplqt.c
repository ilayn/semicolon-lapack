/**
 * @file dtplqt.c
 * @brief DTPLQT computes a blocked LQ factorization of a real
 *        "triangular-pentagonal" matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DTPLQT computes a blocked LQ factorization of a real
 * "triangular-pentagonal" matrix C, which is composed of a
 * triangular block A and pentagonal block B, using the compact
 * WY representation for Q.
 *
 * @param[in]     m     The number of rows of the matrix B, and the order of
 *                      the triangular matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix B. n >= 0.
 * @param[in]     l     The number of rows of the lower trapezoidal part of B.
 *                      min(m,n) >= l >= 0. See Further Details.
 * @param[in]     mb    The block size to be used in the blocked LQ. m >= mb >= 1.
 * @param[in,out] A     Double precision array, dimension (lda,m).
 *                      On entry, the lower triangular m-by-m matrix A.
 *                      On exit, the elements on and below the diagonal contain
 *                      the lower triangular matrix L.
 * @param[in]     lda   The leading dimension of A. lda >= max(1,m).
 * @param[in,out] B     Double precision array, dimension (ldb,n).
 *                      On entry, the pentagonal m-by-n matrix B. The first n-l
 *                      columns are rectangular, and the last l columns are lower
 *                      trapezoidal.
 *                      On exit, B contains the pentagonal matrix V.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1,m).
 * @param[out]    T     Double precision array, dimension (ldt,m).
 *                      The lower triangular block reflectors stored in compact
 *                      form as a sequence of upper triangular blocks.
 * @param[in]     ldt   The leading dimension of T. ldt >= mb.
 * @param[out]    work  Double precision array, dimension (mb*m).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dtplqt(const int m, const int n, const int l, const int mb,
            f64* const restrict A, const int lda,
            f64* const restrict B, const int ldb,
            f64* const restrict T, const int ldt,
            f64* const restrict work, int* info)
{
    int i, ib, lb, nb, iinfo;
    int minmn;

    *info = 0;
    minmn = m < n ? m : n;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (l < 0 || (l > minmn && minmn >= 0)) {
        *info = -3;
    } else if (mb < 1 || (mb > m && m > 0)) {
        *info = -4;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -6;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldt < mb) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DTPLQT", -(*info));
        return;
    }

    if (m == 0 || n == 0) return;

    for (i = 0; i < m; i += mb) {
        ib = ((m - i) < mb) ? (m - i) : mb;
        nb = ((n - l + i + ib) < n) ? (n - l + i + ib) : n;
        if (i >= l) {
            lb = 0;
        } else {
            lb = nb - n + l - i;
        }

        dtplqt2(ib, nb, lb, &A[i + i * lda], lda, &B[i], ldb,
                &T[i * ldt], ldt, &iinfo);

        if (i + ib < m) {
            dtprfb("R", "N", "F", "R", m - i - ib, nb, ib, lb,
                   &B[i], ldb, &T[i * ldt], ldt,
                   &A[(i + ib) + i * lda], lda, &B[i + ib], ldb,
                   work, m - i - ib);
        }
    }
}
