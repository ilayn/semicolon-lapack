/**
 * @file ztplqt2.c
 * @brief ZTPLQT2 computes a LQ factorization of a complex "triangular-pentagonal"
 *        matrix, which is composed of a triangular block and a pentagonal block,
 *        using the compact WY representation for Q.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTPLQT2 computes a LQ factorization of a complex "triangular-pentagonal"
 * matrix C, which is composed of a triangular block A and pentagonal block B,
 * using the compact WY representation for Q.
 *
 * @param[in]     m     The total number of rows of the matrix B. m >= 0.
 * @param[in]     n     The number of columns of the matrix B, and the order of
 *                      the triangular matrix A. n >= 0.
 * @param[in]     l     The number of rows of the lower trapezoidal part of B.
 *                      min(m,n) >= l >= 0. See Further Details.
 * @param[in,out] A     Double complex array, dimension (lda,m).
 *                      On entry, the lower triangular m-by-m matrix A.
 *                      On exit, the elements on and below the diagonal contain
 *                      the lower triangular matrix L.
 * @param[in]     lda   The leading dimension of A. lda >= max(1,m).
 * @param[in,out] B     Double complex array, dimension (ldb,n).
 *                      On entry, the pentagonal m-by-n matrix B. The first n-l
 *                      columns are rectangular, and the last l columns are lower
 *                      trapezoidal.
 *                      On exit, B contains the pentagonal matrix V.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1,m).
 * @param[out]    T     Double complex array, dimension (ldt,m).
 *                      The n-by-n upper triangular factor T of the block
 *                      reflector. See Further Details.
 * @param[in]     ldt   The leading dimension of T. ldt >= max(1,m).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ztplqt2(const INT m, const INT n, const INT l,
             c128* restrict A, const INT lda,
             c128* restrict B, const INT ldb,
             c128* restrict T, const INT ldt, INT* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);
    INT i, j, p, mp, np;
    INT minmn;
    c128 alpha;

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
        xerbla("ZTPLQT2", -(*info));
        return;
    }

    if (n == 0 || m == 0) return;

    for (i = 0; i < m; i++) {

        p = n - l + (l < (i + 1) ? l : (i + 1));

        zlarfg(p + 1, &A[i + i * lda], &B[i], ldb, &T[i * ldt]);
        T[i * ldt] = conj(T[i * ldt]);
        if (i < m - 1) {
            for (j = 0; j < p; j++) {
                B[i + j * ldb] = conj(B[i + j * ldb]);
            }

            for (j = 0; j < m - i - 1; j++) {
                T[(m - 1) + j * ldt] = A[(i + 1 + j) + i * lda];
            }
            cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, p, &ONE,
                        &B[i + 1], ldb, &B[i], ldb, &ONE,
                        &T[m - 1], ldt);

            alpha = -T[i * ldt];
            for (j = 0; j < m - i - 1; j++) {
                A[(i + 1 + j) + i * lda] = A[(i + 1 + j) + i * lda] +
                                           alpha * T[(m - 1) + j * ldt];
            }
            cblas_zgerc(CblasColMajor, m - i - 1, p, &alpha, &T[m - 1], ldt,
                        &B[i], ldb, &B[i + 1], ldb);
            for (j = 0; j < p; j++) {
                B[i + j * ldb] = conj(B[i + j * ldb]);
            }
        }
    }

    for (i = 1; i < m; i++) {
        alpha = -T[i * ldt];

        for (j = 0; j < i; j++) {
            T[i + j * ldt] = ZERO;
        }
        p = (i < l) ? i : l;
        np = (n - l + 1) < n ? (n - l + 1) : n;
        mp = (p + 1) < m ? (p + 1) : m;
        for (j = 0; j < n - l + p; j++) {
            B[i + j * ldb] = conj(B[i + j * ldb]);
        }

        for (j = 0; j < p; j++) {
            T[i + j * ldt] = alpha * B[i + (np - 1 + j) * ldb];
        }
        cblas_ztrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    p, &B[(np - 1) * ldb], ldb, &T[i], ldt);

        if (i - p > 0) {
            cblas_zgemv(CblasColMajor, CblasNoTrans, i - p, l, &alpha,
                        &B[(mp - 1) + (np - 1) * ldb], ldb, &B[i + (np - 1) * ldb], ldb, &ZERO,
                        &T[i + (mp - 1) * ldt], ldt);
        }

        if (n - l > 0) {
            cblas_zgemv(CblasColMajor, CblasNoTrans, i, n - l, &alpha,
                        B, ldb, &B[i], ldb, &ONE, &T[i], ldt);
        }

        for (j = 0; j < i; j++) {
            T[i + j * ldt] = conj(T[i + j * ldt]);
        }
        cblas_ztrmv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                    i, T, ldt, &T[i], ldt);
        for (j = 0; j < i; j++) {
            T[i + j * ldt] = conj(T[i + j * ldt]);
        }
        for (j = 0; j < n - l + p; j++) {
            B[i + j * ldb] = conj(B[i + j * ldb]);
        }

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
