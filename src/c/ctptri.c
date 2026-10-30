/**
 * @file ctptri.c
 * @brief CTPTRI computes the inverse of a triangular matrix stored in packed format.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CTPTRI computes the inverse of a complex upper or lower triangular
 * matrix A stored in packed format.
 *
 * @param[in]     uplo   = 'U': A is upper triangular;
 *                        = 'L': A is lower triangular.
 * @param[in]     diag   = 'N': A is non-unit triangular;
 *                        = 'U': A is unit triangular.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangular matrix A,
 *                       stored columnwise in a linear array.
 *                       On exit, the (triangular) inverse of the original
 *                       matrix, in the same packed storage format.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, A(i,i) is exactly zero. The triangular
 *                           matrix is singular and its inverse cannot be computed.
 */
void ctptri(
    const char* uplo,
    const char* diag,
    const INT n,
    c64* restrict AP,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    // Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    INT nounit = (diag[0] == 'N' || diag[0] == 'n');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("CTPTRI", -(*info));
        return;
    }

    // Check for singularity if non-unit
    if (nounit) {
        if (upper) {
            INT jj = -1;
            for (INT i = 0; i < n; i++) {
                jj = jj + (i + 1);
                if (AP[jj] == ZERO) {
                    *info = i + 1;
                    return;
                }
            }
        } else {
            INT jj = 0;
            for (INT i = 0; i < n; i++) {
                if (AP[jj] == ZERO) {
                    *info = i + 1;
                    return;
                }
                jj = jj + n - i;
            }
        }
        *info = 0;
    }

    CBLAS_DIAG cblas_siag = nounit ? CblasNonUnit : CblasUnit;

    if (upper) {
        // Compute inverse of upper triangular matrix
        INT jc = 0;
        for (INT j = 0; j < n; j++) {
            c64 ajj;
            if (nounit) {
                AP[jc + j] = ONE / AP[jc + j];
                ajj = -AP[jc + j];
            } else {
                ajj = -ONE;
            }

            // Compute elements 0:j-1 of j-th column
            if (j > 0) {
                cblas_ctpmv(CblasColMajor, CblasUpper, CblasNoTrans, cblas_siag,
                            j, AP, &AP[jc], 1);
                cblas_cscal(j, &ajj, &AP[jc], 1);
            }
            jc = jc + (j + 1);
        }
    } else {
        // Compute inverse of lower triangular matrix
        INT jc = n * (n + 1) / 2 - 1;
        INT jclast = 0;
        for (INT j = n - 1; j >= 0; j--) {
            c64 ajj;
            if (nounit) {
                AP[jc] = ONE / AP[jc];
                ajj = -AP[jc];
            } else {
                ajj = -ONE;
            }
            if (j < n - 1) {
                // Compute elements j+1:n-1 of j-th column
                cblas_ctpmv(CblasColMajor, CblasLower, CblasNoTrans, cblas_siag,
                            n - j - 1, &AP[jclast], &AP[jc + 1], 1);
                cblas_cscal(n - j - 1, &ajj, &AP[jc + 1], 1);
            }
            jclast = jc;
            jc = jc - (n - j + 1);
        }
    }
}
