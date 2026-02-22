/**
 * @file clacrm.c
 * @brief CLACRM multiplies a complex matrix by a square real matrix.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * CLACRM performs a very simple matrix-matrix multiplication:
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
 * @param[in]     B       Single precision array, dimension (ldb, n).
 *                        On entry, B contains the N by N matrix B.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    C       Complex array, dimension (ldc, n).
 *                        On exit, C contains the M by N matrix C.
 * @param[in]     ldc     The leading dimension of the array C. ldc >= max(1,m).
 * @param[out]    rwork   Single precision array, dimension (2*m*n).
 */
void clacrm(
    const INT m,
    const INT n,
    const c64* restrict A,
    const INT lda,
    const f32* restrict B,
    const INT ldb,
    c64* restrict C,
    const INT ldc,
    f32* restrict rwork)
{
    INT i, j, l;

    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    if ((m == 0) || (n == 0)) {
        return;
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = crealf(A[i + j * lda]);
        }
    }

    l = m * n;
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, ONE, rwork, m, B, ldb, ZERO,
                &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLXF(rwork[l + j * m + i], 0.0f);
        }
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = cimagf(A[i + j * lda]);
        }
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, ONE, rwork, m, B, ldb, ZERO,
                &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLXF(crealf(C[i + j * ldc]),
                                    rwork[l + j * m + i]);
        }
    }
}
