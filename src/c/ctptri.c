/**
 * @file ctptri.c
 * @brief CTPTRI computes the inverse of a triangular matrix stored in packed format.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
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
    // ctptri.f lines 132-133: Parameters
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    // ctptri.f lines 151-164: Test the input parameters
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

    // ctptri.f lines 168-185: Check for singularity if non-unit
    if (nounit) {
        if (upper) {
            // ctptri.f lines 169-175
            INT jj = -1;  // 0-based: start at -1 so first jj += (i+1) gives 0
            for (INT i = 0; i < n; i++) {
                jj = jj + (i + 1);  // ctptri.f line 172: JJ = JJ + INFO
                if (AP[jj] == ZERO) {
                    *info = i + 1;  // 1-based error code
                    return;
                }
            }
        } else {
            // ctptri.f lines 176-183
            INT jj = 0;  // ctptri.f line 177: JJ = 1 (0-based: 0)
            for (INT i = 0; i < n; i++) {
                if (AP[jj] == ZERO) {
                    *info = i + 1;  // 1-based error code
                    return;
                }
                jj = jj + n - i;  // ctptri.f line 181: JJ = JJ + N - INFO + 1
            }
        }
        *info = 0;  // ctptri.f line 184
    }

    // Prepare CBLAS enum for diag
    CBLAS_DIAG cblas_siag = nounit ? CblasNonUnit : CblasUnit;

    if (upper) {
        // ctptri.f lines 187-206: Compute inverse of upper triangular matrix
        INT jc = 0;  // ctptri.f line 191: JC = 1 (0-based: 0)
        for (INT j = 0; j < n; j++) {  // ctptri.f line 192: DO 30 J = 1, N
            c64 ajj;
            if (nounit) {
                // ctptri.f lines 193-195
                AP[jc + j] = ONE / AP[jc + j];
                ajj = -AP[jc + j];
            } else {
                // ctptri.f lines 196-198
                ajj = -ONE;
            }

            // ctptri.f lines 202-204: Compute elements 1:j-1 of j-th column
            if (j > 0) {
                cblas_ctpmv(CblasColMajor, CblasUpper, CblasNoTrans, cblas_siag,
                            j, AP, &AP[jc], 1);
                cblas_cscal(j, &ajj, &AP[jc], 1);
            }
            jc = jc + (j + 1);  // ctptri.f line 205: JC = JC + J
        }
    } else {
        // ctptri.f lines 208-231: Compute inverse of lower triangular matrix
        INT jc = n * (n + 1) / 2 - 1;  // ctptri.f line 212: JC = N*(N+1)/2 (0-based: subtract 1)
        INT jclast = 0;  // Will be set in loop
        for (INT j = n - 1; j >= 0; j--) {  // ctptri.f line 213: DO 40 J = N, 1, -1
            c64 ajj;
            if (nounit) {
                // ctptri.f lines 214-216
                AP[jc] = ONE / AP[jc];
                ajj = -AP[jc];
            } else {
                // ctptri.f lines 217-219
                ajj = -ONE;
            }
            if (j < n - 1) {
                // ctptri.f lines 220-226: Compute elements j+1:n of j-th column
                cblas_ctpmv(CblasColMajor, CblasLower, CblasNoTrans, cblas_siag,
                            n - j - 1, &AP[jclast], &AP[jc + 1], 1);
                cblas_cscal(n - j - 1, &ajj, &AP[jc + 1], 1);
            }
            jclast = jc;  // ctptri.f line 228: JCLAST = JC
            // ctptri.f line 229: JC = JC - N + J - 2
            jc = jc - (n - j + 1);  // Move to previous column start
        }
    }
}
