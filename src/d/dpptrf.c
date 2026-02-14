/**
 * @file dpptrf.c
 * @brief DPPTRF computes the Cholesky factorization of a packed symmetric positive definite matrix.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPPTRF computes the Cholesky factorization of a real symmetric
 * positive definite matrix A stored in packed format.
 *
 * The factorization has the form
 *    A = U**T * U,  if UPLO = 'U', or
 *    A = L  * L**T,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, if info = 0, the triangular factor U or L from
 *                       the Cholesky factorization A = U**T*U or A = L*L**T,
 *                       in the same storage format as A.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the leading principal minor of order i
 *                           is not positive, and the factorization could not be
 *                           completed.
 */
void dpptrf(
    const char* uplo,
    const int n,
    f64* const restrict AP,
    int* info)
{
    // dpptrf.f lines 134-135: Parameters
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // dpptrf.f lines 157-167: Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("DPPTRF", -(*info));
        return;
    }

    // dpptrf.f lines 171-172: Quick return if possible
    if (n == 0) {
        return;
    }

    if (upper) {
        // dpptrf.f lines 174-197: Compute the Cholesky factorization A = U**T*U.
        int jj = -1;  // dpptrf.f line 178: JJ = 0 (0-based: jj starts at -1 so first jj+j = 0)
        for (int j = 0; j < n; j++) {  // dpptrf.f line 179: DO 10 J = 1, N
            int jc = jj + 1;  // dpptrf.f line 180: JC = JJ + 1 (0-based: jc = jj + 1)
            jj = jj + (j + 1);  // dpptrf.f line 181: JJ = JJ + J (0-based: j+1)

            // dpptrf.f lines 185-187: Compute elements 1:J-1 of column J.
            if (j > 0) {
                cblas_dtpsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            j, AP, &AP[jc], 1);
            }

            // dpptrf.f lines 191-196: Compute U(J,J) and test for non-positive-definiteness.
            f64 ajj = AP[jj];
            if (j > 0) {
                ajj -= cblas_ddot(j, &AP[jc], 1, &AP[jc], 1);
            }
            if (ajj <= ZERO) {
                AP[jj] = ajj;
                *info = j + 1;  // 1-based error code
                return;
            }
            AP[jj] = sqrt(ajj);
        }
    } else {
        // dpptrf.f lines 198-225: Compute the Cholesky factorization A = L*L**T.
        int jj = 0;  // dpptrf.f line 202: JJ = 1 (0-based: jj = 0)
        for (int j = 0; j < n; j++) {  // dpptrf.f line 203: DO 20 J = 1, N

            // dpptrf.f lines 207-213: Compute L(J,J) and test for non-positive-definiteness.
            f64 ajj = AP[jj];
            if (ajj <= ZERO) {
                AP[jj] = ajj;
                *info = j + 1;  // 1-based error code
                return;
            }
            ajj = sqrt(ajj);
            AP[jj] = ajj;

            // dpptrf.f lines 218-223: Compute elements J+1:N of column J and update
            // the trailing submatrix.
            if (j < n - 1) {
                // dpptrf.f line 219: CALL DSCAL( N-J, ONE / AJJ, AP( JJ+1 ), 1 )
                cblas_dscal(n - j - 1, ONE / ajj, &AP[jj + 1], 1);
                // dpptrf.f lines 220-221: CALL DSPR( 'Lower', N-J, -ONE, AP( JJ+1 ), 1, AP( JJ+N-J+1 ) )
                // In 0-based: JJ+N-J+1 becomes jj + (n - j) = jj + n - j
                // But wait: Fortran JJ+N-J+1 with 1-based J means position of next diagonal
                // In 0-based: jj + (n - j) is correct for the start of next column
                cblas_dspr(CblasColMajor, CblasLower, n - j - 1, -ONE,
                           &AP[jj + 1], 1, &AP[jj + n - j]);
                // dpptrf.f line 222: JJ = JJ + N - J + 1
                jj = jj + n - j;  // 0-based: jj + (n - (j+1) + 1) = jj + n - j
            }
        }
    }
}
