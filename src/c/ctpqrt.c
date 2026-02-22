/**
 * @file ctpqrt.c
 * @brief CTPQRT computes a blocked QR factorization of a complex
 *        "triangular-pentagonal" matrix.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CTPQRT computes a blocked QR factorization of a complex
 * "triangular-pentagonal" matrix C, which is composed of a
 * triangular block A and pentagonal block B, using the compact
 * WY representation for Q.
 *
 * @param[in]     m     The number of rows of the matrix B. m >= 0.
 * @param[in]     n     The number of columns of the matrix B, and the order of
 *                      the triangular matrix A. n >= 0.
 * @param[in]     l     The number of rows of the upper trapezoidal part of B.
 *                      min(m,n) >= l >= 0. See Further Details.
 * @param[in]     nb    The block size to be used in the blocked QR. n >= nb >= 1.
 * @param[in,out] A     Single complex array, dimension (lda,n).
 *                      On entry, the upper triangular n-by-n matrix A.
 *                      On exit, the elements on and above the diagonal contain
 *                      the upper triangular matrix R.
 * @param[in]     lda   The leading dimension of A. lda >= max(1,n).
 * @param[in,out] B     Single complex array, dimension (ldb,n).
 *                      On entry, the pentagonal m-by-n matrix B. The first m-l
 *                      rows are rectangular, and the last l rows are upper
 *                      trapezoidal.
 *                      On exit, B contains the pentagonal matrix V.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1,m).
 * @param[out]    T     Single complex array, dimension (ldt,n).
 *                      The upper triangular block reflectors stored in compact
 *                      form as a sequence of upper triangular blocks.
 * @param[in]     ldt   The leading dimension of T. ldt >= nb.
 * @param[out]    work  Single complex array, dimension (nb*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ctpqrt(const INT m, const INT n, const INT l, const INT nb,
            c64* restrict A, const INT lda,
            c64* restrict B, const INT ldb,
            c64* restrict T, const INT ldt,
            c64* restrict work, INT* info)
{
    INT i, ib, lb, mb, iinfo;
    INT minmn;

    *info = 0;
    minmn = m < n ? m : n;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (l < 0 || (l > minmn && minmn >= 0)) {
        *info = -3;
    } else if (nb < 1 || (nb > n && n > 0)) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldt < nb) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("CTPQRT", -(*info));
        return;
    }

    if (m == 0 || n == 0) return;

    for (i = 0; i < n; i += nb) {
        ib = ((n - i) < nb) ? (n - i) : nb;
        mb = ((m - l + i + ib) < m) ? (m - l + i + ib) : m;
        if (i + 1 >= l) {
            lb = 0;
        } else {
            lb = mb - m + l - i;
        }

        ctpqrt2(mb, ib, lb, &A[i + i * lda], lda, &B[i * ldb], ldb,
                &T[i * ldt], ldt, &iinfo);

        if (i + ib < n) {
            ctprfb("L", "C", "F", "C", mb, n - i - ib, ib, lb,
                   &B[i * ldb], ldb, &T[i * ldt], ldt,
                   &A[i + (i + ib) * lda], lda, &B[(i + ib) * ldb], ldb,
                   work, ib);
        }
    }
}
