/**
 * @file ctpqrt2.c
 * @brief CTPQRT2 computes a QR factorization of a complex "triangular-pentagonal"
 *        matrix, which is composed of a triangular block and a pentagonal block,
 *        using the compact WY representation for Q.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CTPQRT2 computes a QR factorization of a complex "triangular-pentagonal"
 * matrix C, which is composed of a triangular block A and pentagonal block B,
 * using the compact WY representation for Q.
 *
 * @param[in]     m     The total number of rows of the matrix B. m >= 0.
 * @param[in]     n     The number of columns of the matrix B, and the order of
 *                      the triangular matrix A. n >= 0.
 * @param[in]     l     The number of rows of the upper trapezoidal part of B.
 *                      min(m,n) >= l >= 0. See Further Details.
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
 *                      The n-by-n upper triangular factor T of the block
 *                      reflector. See Further Details.
 * @param[in]     ldt   The leading dimension of T. ldt >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ctpqrt2(const INT m, const INT n, const INT l,
             c64* restrict A, const INT lda,
             c64* restrict B, const INT ldb,
             c64* restrict T, const INT ldt, INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    INT i, j, p, mp, np;
    INT minmn;
    c64 alpha;

    *info = 0;
    minmn = m < n ? m : n;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (l < 0 || l > minmn) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -7;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("CTPQRT2", -(*info));
        return;
    }

    if (n == 0 || m == 0) return;

    for (i = 0; i < n; i++) {
        p = m - l + (l < (i + 1) ? l : (i + 1));  /* M-L+MIN(L, I+1) in 0-based */

        clarfg(p + 1, &A[i + i * lda], &B[i * ldb], 1, &T[i]);

        if (i < n - 1) {
            for (j = 0; j < n - i - 1; j++) {
                T[j + (n - 1) * ldt] = conjf(A[i + (i + 1 + j) * lda]);
            }
            cblas_cgemv(CblasColMajor, CblasConjTrans, p, n - i - 1, &ONE,
                        &B[(i + 1) * ldb], ldb, &B[i * ldb], 1, &ONE,
                        &T[(n - 1) * ldt], 1);

            alpha = -conjf(T[i]);
            for (j = 0; j < n - i - 1; j++) {
                A[i + (i + 1 + j) * lda] = A[i + (i + 1 + j) * lda] +
                                           alpha * conjf(T[j + (n - 1) * ldt]);
            }
            cblas_cgerc(CblasColMajor, p, n - i - 1, &alpha, &B[i * ldb], 1,
                       &T[(n - 1) * ldt], 1, &B[(i + 1) * ldb], ldb);
        }
    }

    for (i = 1; i < n; i++) {
        alpha = -T[i];

        for (j = 0; j < i; j++) {
            T[j + i * ldt] = ZERO;
        }
        p = (i < l) ? i : l;                     /* MIN(I, L) in 0-based: i is already 0-based */
        mp = m - l;                               /* 0-based index of first row of B2 */
        np = (p + 1) < n ? (p + 1) : n;          /* MIN(P+1, N) */

        for (j = 0; j < p; j++) {
            T[j + i * ldt] = alpha * B[(mp + j) + i * ldb];
        }
        cblas_ctrmv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                    p, &B[mp], ldb, &T[i * ldt], 1);

        if (i - p > 0) {
            cblas_cgemv(CblasColMajor, CblasConjTrans, l, i - p, &alpha,
                        &B[mp + (np - 1) * ldb], ldb, &B[mp + i * ldb], 1, &ZERO,
                        &T[(np - 1) + i * ldt], 1);
        }

        if (m - l > 0) {
            cblas_cgemv(CblasColMajor, CblasConjTrans, m - l, i, &alpha,
                        B, ldb, &B[i * ldb], 1, &ONE, &T[i * ldt], 1);
        }

        cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    i, T, ldt, &T[i * ldt], 1);

        T[i + i * ldt] = T[i];
        T[i] = ZERO;
    }
}
