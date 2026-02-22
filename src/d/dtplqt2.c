/**
 * @file dtplqt2.c
 * @brief DTPLQT2 computes a LQ factorization of a real "triangular-pentagonal"
 *        matrix, which is composed of a triangular block and a pentagonal block,
 *        using the compact WY representation for Q.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DTPLQT2 computes a LQ factorization of a real "triangular-pentagonal"
 * matrix C, which is composed of a triangular block A and pentagonal block B,
 * using the compact WY representation for Q.
 *
 * @param[in]     m     The total number of rows of the matrix B. m >= 0.
 * @param[in]     n     The number of columns of the matrix B, and the order of
 *                      the triangular matrix A. n >= 0.
 * @param[in]     l     The number of rows of the lower trapezoidal part of B.
 *                      min(m,n) >= l >= 0. See Further Details.
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
 *                      The n-by-n upper triangular factor T of the block
 *                      reflector. See Further Details.
 * @param[in]     ldt   The leading dimension of T. ldt >= max(1,m).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dtplqt2(const INT m, const INT n, const INT l,
             f64* restrict A, const INT lda,
             f64* restrict B, const INT ldb,
             f64* restrict T, const INT ldt, INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    INT i, j, p, mp, np;
    INT minmn;
    f64 alpha;

    *info = 0;
    minmn = m < n ? m : n;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (l < 0 || l > minmn) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -7;
    } else if (ldt < (m > 1 ? m : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DTPLQT2", -(*info));
        return;
    }

    if (n == 0 || m == 0) return;

    for (i = 0; i < m; i++) {
        p = n - l + (l < (i + 1) ? l : (i + 1));  /* N-L+MIN(L, I+1) in 0-based */

        dlarfg(p + 1, &A[i + i * lda], &B[i], ldb, &T[i * ldt]);

        if (i < m - 1) {
            for (j = 0; j < m - i - 1; j++) {
                T[(m - 1) + j * ldt] = A[(i + 1 + j) + i * lda];
            }
            cblas_dgemv(CblasColMajor, CblasNoTrans, m - i - 1, p, ONE,
                        &B[i + 1], ldb, &B[i], ldb, ONE,
                        &T[m - 1], ldt);

            alpha = -T[i * ldt];
            for (j = 0; j < m - i - 1; j++) {
                A[(i + 1 + j) + i * lda] = A[(i + 1 + j) + i * lda] +
                                           alpha * T[(m - 1) + j * ldt];
            }
            cblas_dger(CblasColMajor, m - i - 1, p, alpha, &T[m - 1], ldt,
                       &B[i], ldb, &B[i + 1], ldb);
        }
    }

    for (i = 1; i < m; i++) {
        alpha = -T[i * ldt];

        for (j = 0; j < i; j++) {
            T[i + j * ldt] = ZERO;
        }
        p = (i < l) ? i : l;                     /* MIN(I, L) */
        np = (n - l + 1) < n ? (n - l + 1) : n;  /* MIN(N-L+1, N) */
        mp = (p + 1) < m ? (p + 1) : m;          /* MIN(P+1, M) */

        for (j = 0; j < p; j++) {
            T[i + j * ldt] = alpha * B[i + (np - 1 + j) * ldb];
        }
        cblas_dtrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    p, &B[(np - 1) * ldb], ldb, &T[i], ldt);

        if (i - p > 0) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, i - p, l, alpha,
                        &B[(mp - 1) + (np - 1) * ldb], ldb, &B[i + (np - 1) * ldb], ldb, ZERO,
                        &T[i + (mp - 1) * ldt], ldt);
        }

        if (n - l > 0) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, i, n - l, alpha,
                        B, ldb, &B[i], ldb, ONE, &T[i], ldt);
        }

        cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    i, T, ldt, &T[i], ldt);

        T[i + i * ldt] = T[i * ldt];
        T[i * ldt] = ZERO;
    }

    for (i = 0; i < m; i++) {
        for (j = i + 1; j < m; j++) {
            T[i + j * ldt] = T[j + i * ldt];
            T[j + i * ldt] = ZERO;
        }
    }
}
