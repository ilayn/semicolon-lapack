/**
 * @file stptri.c
 * @brief STPTRI computes the inverse of a triangular matrix stored in packed format.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STPTRI computes the inverse of a real upper or lower triangular
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
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 *                       > 0: if info = i, A(i,i) is exactly zero. The triangular
 *                            matrix is singular and its inverse cannot be computed.
 */
void stptri(
    const char* uplo,
    const char* diag,
    const int n,
    float* const restrict AP,
    int* info)
{
    // stptri.f lines 132-133: Parameters
    const float ONE = 1.0f;
    const float ZERO = 0.0f;

    // stptri.f lines 151-164: Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    int nounit = (diag[0] == 'N' || diag[0] == 'n');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("STPTRI", -(*info));
        return;
    }

    // stptri.f lines 168-185: Check for singularity if non-unit
    if (nounit) {
        if (upper) {
            // stptri.f lines 169-175
            int jj = -1;  // 0-based: start at -1 so first jj += (i+1) gives 0
            for (int i = 0; i < n; i++) {
                jj = jj + (i + 1);  // stptri.f line 172: JJ = JJ + INFO
                if (AP[jj] == ZERO) {
                    *info = i + 1;  // 1-based error code
                    return;
                }
            }
        } else {
            // stptri.f lines 176-183
            int jj = 0;  // stptri.f line 177: JJ = 1 (0-based: 0)
            for (int i = 0; i < n; i++) {
                if (AP[jj] == ZERO) {
                    *info = i + 1;  // 1-based error code
                    return;
                }
                jj = jj + n - i;  // stptri.f line 181: JJ = JJ + N - INFO + 1
            }
        }
        *info = 0;  // stptri.f line 184
    }

    // Prepare CBLAS enum for diag
    CBLAS_DIAG cblas_siag = nounit ? CblasNonUnit : CblasUnit;

    if (upper) {
        // stptri.f lines 187-206: Compute inverse of upper triangular matrix
        int jc = 0;  // stptri.f line 191: JC = 1 (0-based: 0)
        for (int j = 0; j < n; j++) {  // stptri.f line 192: DO 30 J = 1, N
            float ajj;
            if (nounit) {
                // stptri.f lines 193-195
                // AP( JC+J-1 ) in 1-based = AP[jc + j] in 0-based (since JC is 1-based offset)
                // Actually: JC is column start (1-based), J-1 is offset to diagonal
                // In 0-based: jc + j is the diagonal element
                AP[jc + j] = ONE / AP[jc + j];
                ajj = -AP[jc + j];
            } else {
                // stptri.f lines 196-198
                ajj = -ONE;
            }

            // stptri.f lines 202-204: Compute elements 1:j-1 of j-th column
            if (j > 0) {
                cblas_stpmv(CblasColMajor, CblasUpper, CblasNoTrans, cblas_siag,
                            j, AP, &AP[jc], 1);
                cblas_sscal(j, ajj, &AP[jc], 1);
            }
            jc = jc + (j + 1);  // stptri.f line 205: JC = JC + J
        }
    } else {
        // stptri.f lines 208-231: Compute inverse of lower triangular matrix
        int jc = n * (n + 1) / 2 - 1;  // stptri.f line 212: JC = N*(N+1)/2 (0-based: subtract 1)
        int jclast = 0;  // Will be set in loop
        for (int j = n - 1; j >= 0; j--) {  // stptri.f line 213: DO 40 J = N, 1, -1
            float ajj;
            if (nounit) {
                // stptri.f lines 214-216
                AP[jc] = ONE / AP[jc];
                ajj = -AP[jc];
            } else {
                // stptri.f lines 217-219
                ajj = -ONE;
            }
            if (j < n - 1) {
                // stptri.f lines 220-226: Compute elements j+1:n of j-th column
                // stptri.f lines 224-225: CALL DTPMV( 'Lower', 'No transpose', DIAG, N-J, AP( JCLAST ), AP( JC+1 ), 1 )
                cblas_stpmv(CblasColMajor, CblasLower, CblasNoTrans, cblas_siag,
                            n - j - 1, &AP[jclast], &AP[jc + 1], 1);
                cblas_sscal(n - j - 1, ajj, &AP[jc + 1], 1);
            }
            jclast = jc;  // stptri.f line 228: JCLAST = JC
            // stptri.f line 229: JC = JC - N + J - 2
            // In Fortran 1-based with J going from N to 1: stride = N - J + 2
            // In C 0-based with j going from n-1 to 0: stride = n - (j+1) + 2 = n - j + 1
            jc = jc - (n - j + 1);  // Move to previous column start
        }
    }
}
