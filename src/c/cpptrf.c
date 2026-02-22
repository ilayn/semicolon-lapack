/**
 * @file cpptrf.c
 * @brief CPPTRF computes the Cholesky factorization of a packed Hermitian positive definite matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPPTRF computes the Cholesky factorization of a complex Hermitian
 * positive definite matrix A stored in packed format.
 *
 * The factorization has the form
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L  * L**H,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, if info = 0, the triangular factor U or L from
 *                       the Cholesky factorization A = U**H*U or A = L*L**H,
 *                       in the same storage format as A.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the leading principal minor of order i
 *                           is not positive, and the factorization could not be
 *                           completed.
 */
void cpptrf(
    const char* uplo,
    const INT n,
    c64* restrict AP,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("CPPTRF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {
        // Compute the Cholesky factorization A = U**H * U.
        INT jj = -1;
        for (INT j = 0; j < n; j++) {
            INT jc = jj + 1;
            jj = jj + (j + 1);

            // Compute elements 1:J-1 of column J.
            if (j > 0) {
                cblas_ctpsv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                            j, AP, &AP[jc], 1);
            }

            // Compute U(J,J) and test for non-positive-definiteness.
            f32 ajj = crealf(AP[jj]);
            if (j > 0) {
                c64 dotc;
                cblas_cdotc_sub(j, &AP[jc], 1, &AP[jc], 1, &dotc);
                ajj -= crealf(dotc);
            }
            if (ajj <= ZERO) {
                AP[jj] = ajj;
                *info = j + 1;
                return;
            }
            AP[jj] = sqrtf(ajj);
        }
    } else {
        // Compute the Cholesky factorization A = L * L**H.
        INT jj = 0;
        for (INT j = 0; j < n; j++) {

            // Compute L(J,J) and test for non-positive-definiteness.
            f32 ajj = crealf(AP[jj]);
            if (ajj <= ZERO) {
                AP[jj] = ajj;
                *info = j + 1;
                return;
            }
            ajj = sqrtf(ajj);
            AP[jj] = ajj;

            // Compute elements J+1:N of column J and update
            // the trailing submatrix.
            if (j < n - 1) {
                cblas_csscal(n - j - 1, ONE / ajj, &AP[jj + 1], 1);
                cblas_chpr(CblasColMajor, CblasLower, n - j - 1, -ONE,
                           &AP[jj + 1], 1, &AP[jj + n - j]);
                jj = jj + n - j;
            }
        }
    }
}
