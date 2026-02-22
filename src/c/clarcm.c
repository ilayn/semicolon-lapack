/**
 * @file clarcm.c
 * @brief CLARCM performs a very simple matrix-matrix multiplication: C := A * B.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CLARCM performs a very simple matrix-matrix multiplication:
 *          C := A * B,
 * where A is M by M and real; B is M by N and complex;
 * C is M by N and complex.
 *
 * @param[in]     m       The number of rows of the matrix A and of the matrix C.
 *                        M >= 0.
 * @param[in]     n       The number of columns and rows of the matrix B and
 *                        the number of columns of the matrix C.
 *                        N >= 0.
 * @param[in]     A       Single precision array, dimension (LDA, M).
 *                        On entry, A contains the M by M matrix A.
 * @param[in]     lda     The leading dimension of the array A. LDA >= max(1,M).
 * @param[in]     B       Complex*16 array, dimension (LDB, N).
 *                        On entry, B contains the M by N matrix B.
 * @param[in]     ldb     The leading dimension of the array B. LDB >= max(1,M).
 * @param[out]    C       Complex*16 array, dimension (LDC, N).
 *                        On exit, C contains the M by N matrix C.
 * @param[in]     ldc     The leading dimension of the array C. LDC >= max(1,M).
 * @param[out]    rwork   Single precision array, dimension (2*M*N).
 */
void clarcm(
    const INT m,
    const INT n,
    const f32* restrict A,
    const INT lda,
    const c64* restrict B,
    const INT ldb,
    c64* restrict C,
    const INT ldc,
    f32* restrict rwork)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    INT i, j, l;

    if ((m == 0) || (n == 0)) {
        return;
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = crealf(B[i + j * ldb]);
        }
    }

    l = m * n;
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, ONE, A, lda, rwork, m, ZERO, &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLXF(rwork[l + j * m + i], 0.0f);
        }
    }

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            rwork[j * m + i] = cimagf(B[i + j * ldb]);
        }
    }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, ONE, A, lda, rwork, m, ZERO, &rwork[l], m);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            C[i + j * ldc] = CMPLXF(crealf(C[i + j * ldc]),
                                    rwork[l + j * m + i]);
        }
    }
}
