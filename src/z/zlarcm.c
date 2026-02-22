/**
 * @file zlarcm.c
 * @brief ZLARCM performs a very simple matrix-matrix multiplication: C := A * B.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * ZLARCM performs a very simple matrix-matrix multiplication:
 *          C := A * B,
 * where A is M by M and real; B is M by N and complex;
 * C is M by N and complex.
 *
 * @param[in]     m       The number of rows of the matrix A and of the matrix C.
 *                        M >= 0.
 * @param[in]     n       The number of columns and rows of the matrix B and
 *                        the number of columns of the matrix C.
 *                        N >= 0.
 * @param[in]     A       Double precision array, dimension (LDA, M).
 *                        On entry, A contains the M by M matrix A.
 * @param[in]     lda     The leading dimension of the array A. LDA >= max(1,M).
 * @param[in]     B       Complex*16 array, dimension (LDB, N).
 *                        On entry, B contains the M by N matrix B.
 * @param[in]     ldb     The leading dimension of the array B. LDB >= max(1,M).
 * @param[out]    C       Complex*16 array, dimension (LDC, N).
 *                        On exit, C contains the M by N matrix C.
 * @param[in]     ldc     The leading dimension of the array C. LDC >= max(1,M).
 * @param[out]    rwork   Double precision array, dimension (2*M*N).
 */
void zlarcm(
    const INT m,
    const INT n,
    const f64* restrict A,
    const INT lda,
    const c128* restrict B,
    const INT ldb,
    c128* restrict C,
    const INT ldc,
    f64* restrict rwork)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    INT i, j, l;

    if ((m == 0) || (n == 0)) {
        return;
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = creal(B[i + j * ldb]);
        }
    }

    l = m * n;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, ONE, A, lda, rwork, m, ZERO, &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLX(rwork[l + j * m + i], 0.0);
        }
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = cimag(B[i + j * ldb]);
        }
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, ONE, A, lda, rwork, m, ZERO, &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLX(creal(C[i + j * ldc]),
                                    rwork[l + j * m + i]);
        }
    }
}
