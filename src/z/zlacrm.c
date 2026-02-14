/**
 * @file zlacrm.c
 * @brief ZLACRM multiplies a complex matrix by a square real matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZLACRM performs a very simple matrix-matrix multiplication:
 *          C := A * B,
 * where A is M by N and complex; B is N by N and real;
 * C is M by N and complex.
 *
 * @param[in]     m       The number of rows of the matrix A and of the matrix C.
 *                        m >= 0.
 * @param[in]     n       The number of columns and rows of the matrix B and
 *                        the number of columns of the matrix C.
 *                        n >= 0.
 * @param[in]     A       Complex array, dimension (lda, n).
 *                        On entry, A contains the M by N matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,m).
 * @param[in]     B       Double precision array, dimension (ldb, n).
 *                        On entry, B contains the N by N matrix B.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    C       Complex array, dimension (ldc, n).
 *                        On exit, C contains the M by N matrix C.
 * @param[in]     ldc     The leading dimension of the array C. ldc >= max(1,m).
 * @param[out]    rwork   Double precision array, dimension (2*m*n).
 */
void zlacrm(
    const int m,
    const int n,
    const c128* const restrict A,
    const int lda,
    const f64* const restrict B,
    const int ldb,
    c128* const restrict C,
    const int ldc,
    f64* const restrict rwork)
{
    int i, j, l;

    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    if ((m == 0) || (n == 0)) {
        return;
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = creal(A[i + j * lda]);
        }
    }

    l = m * n;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, ONE, rwork, m, B, ldb, ZERO,
                &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLX(rwork[l + j * m + i], 0.0);
        }
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = cimag(A[i + j * lda]);
        }
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, ONE, rwork, m, B, ldb, ZERO,
                &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLX(creal(C[i + j * ldc]),
                                    rwork[l + j * m + i]);
        }
    }
}
