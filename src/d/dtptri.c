/**
 * @file dtptri.c
 * @brief DTPTRI computes the inverse of a triangular matrix stored in packed format.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DTPTRI computes the inverse of a real upper or lower triangular
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
void dtptri(
    const char* uplo,
    const char* diag,
    const INT n,
    f64* restrict AP,
    INT* info)
{
    // dtptri.f lines 132-133: Parameters
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // dtptri.f lines 151-164: Test the input parameters
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
        xerbla("DTPTRI", -(*info));
        return;
    }

    // dtptri.f lines 168-185: Check for singularity if non-unit
    if (nounit) {
        if (upper) {
            // dtptri.f lines 169-175
            INT jj = -1;  // 0-based: start at -1 so first jj += (i+1) gives 0
            for (INT i = 0; i < n; i++) {
                jj = jj + (i + 1);  // dtptri.f line 172: JJ = JJ + INFO
                if (AP[jj] == ZERO) {
                    *info = i + 1;  // 1-based error code
                    return;
                }
            }
        } else {
            // dtptri.f lines 176-183
            INT jj = 0;  // dtptri.f line 177: JJ = 1 (0-based: 0)
            for (INT i = 0; i < n; i++) {
                if (AP[jj] == ZERO) {
                    *info = i + 1;  // 1-based error code
                    return;
                }
                jj = jj + n - i;  // dtptri.f line 181: JJ = JJ + N - INFO + 1
            }
        }
        *info = 0;  // dtptri.f line 184
    }

    // Prepare CBLAS enum for diag
    CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;

    if (upper) {
        // dtptri.f lines 187-206: Compute inverse of upper triangular matrix
        INT jc = 0;  // dtptri.f line 191: JC = 1 (0-based: 0)
        for (INT j = 0; j < n; j++) {  // dtptri.f line 192: DO 30 J = 1, N
            f64 ajj;
            if (nounit) {
                // dtptri.f lines 193-195
                // AP( JC+J-1 ) in 1-based = AP[jc + j] in 0-based (since JC is 1-based offset)
                // Actually: JC is column start (1-based), J-1 is offset to diagonal
                // In 0-based: jc + j is the diagonal element
                AP[jc + j] = ONE / AP[jc + j];
                ajj = -AP[jc + j];
            } else {
                // dtptri.f lines 196-198
                ajj = -ONE;
            }

            // dtptri.f lines 202-204: Compute elements 1:j-1 of j-th column
            if (j > 0) {
                cblas_dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, cblas_diag,
                            j, AP, &AP[jc], 1);
                cblas_dscal(j, ajj, &AP[jc], 1);
            }
            jc = jc + (j + 1);  // dtptri.f line 205: JC = JC + J
        }
    } else {
        // dtptri.f lines 208-231: Compute inverse of lower triangular matrix
        INT jc = n * (n + 1) / 2 - 1;  // dtptri.f line 212: JC = N*(N+1)/2 (0-based: subtract 1)
        INT jclast = 0;  // Will be set in loop
        for (INT j = n - 1; j >= 0; j--) {  // dtptri.f line 213: DO 40 J = N, 1, -1
            f64 ajj;
            if (nounit) {
                // dtptri.f lines 214-216
                AP[jc] = ONE / AP[jc];
                ajj = -AP[jc];
            } else {
                // dtptri.f lines 217-219
                ajj = -ONE;
            }
            if (j < n - 1) {
                // dtptri.f lines 220-226: Compute elements j+1:n of j-th column
                // dtptri.f lines 224-225: CALL DTPMV( 'Lower', 'No transpose', DIAG, N-J, AP( JCLAST ), AP( JC+1 ), 1 )
                cblas_dtpmv(CblasColMajor, CblasLower, CblasNoTrans, cblas_diag,
                            n - j - 1, &AP[jclast], &AP[jc + 1], 1);
                cblas_dscal(n - j - 1, ajj, &AP[jc + 1], 1);
            }
            jclast = jc;  // dtptri.f line 228: JCLAST = JC
            // dtptri.f line 229: JC = JC - N + J - 2
            // In Fortran 1-based with J going from N to 1: stride = N - J + 2
            // In C 0-based with j going from n-1 to 0: stride = n - (j+1) + 2 = n - j + 1
            jc = jc - (n - j + 1);  // Move to previous column start
        }
    }
}
