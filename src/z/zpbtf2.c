/**
 * @file zpbtf2.c
 * @brief ZPBTF2 computes the Cholesky factorization of a Hermitian positive definite band matrix (unblocked).
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBTF2 computes the Cholesky factorization of a complex Hermitian
 * positive definite band matrix A.
 *
 * The factorization has the form
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L * L**H,  if UPLO = 'L',
 * where U is an upper triangular matrix, U**H is the conjugate transpose
 * of U, and L is lower triangular.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo   = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, the factor U or L.
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           is not positive definite.
 */
void zpbtf2(
    const char* uplo,
    const int n,
    const int kd,
    double complex* const restrict AB,
    const int ldab,
    int* info)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int upper;
    int j, kld, kn;
    double ajj;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (ldab < kd + 1) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("ZPBTF2", -(*info));
        return;
    }

    if (n == 0)
        return;

    kld = (1 > ldab - 1) ? 1 : (ldab - 1);

    if (upper) {
        // Compute the Cholesky factorization A = U**H * U
        for (j = 0; j < n; j++) {
            // Compute U(j,j) and test for non-positive-definiteness
            ajj = creal(AB[kd + j * ldab]);
            if (ajj <= ZERO) {
                AB[kd + j * ldab] = CMPLX(ajj, 0.0);
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            AB[kd + j * ldab] = CMPLX(ajj, 0.0);

            // Compute elements j+1:j+kn of row j and update trailing submatrix
            kn = (kd < n - j - 1) ? kd : (n - j - 1);
            if (kn > 0) {
                cblas_zdscal(kn, ONE / ajj, &AB[kd - 1 + (j + 1) * ldab], kld);
                zlacgv(kn, &AB[kd - 1 + (j + 1) * ldab], kld);
                cblas_zher(CblasColMajor, CblasUpper, kn, -ONE,
                           &AB[kd - 1 + (j + 1) * ldab], kld,
                           &AB[kd + (j + 1) * ldab], kld);
                zlacgv(kn, &AB[kd - 1 + (j + 1) * ldab], kld);
            }
        }
    } else {
        // Compute the Cholesky factorization A = L * L**H
        for (j = 0; j < n; j++) {
            // Compute L(j,j) and test for non-positive-definiteness
            ajj = creal(AB[0 + j * ldab]);
            if (ajj <= ZERO) {
                AB[0 + j * ldab] = CMPLX(ajj, 0.0);
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            AB[0 + j * ldab] = CMPLX(ajj, 0.0);

            // Compute elements j+1:j+kn of column j and update trailing submatrix
            kn = (kd < n - j - 1) ? kd : (n - j - 1);
            if (kn > 0) {
                cblas_zdscal(kn, ONE / ajj, &AB[1 + j * ldab], 1);
                cblas_zher(CblasColMajor, CblasLower, kn, -ONE,
                           &AB[1 + j * ldab], 1,
                           &AB[0 + (j + 1) * ldab], kld);
            }
        }
    }
}
