/**
 * @file cpbstf.c
 * @brief CPBSTF computes a split Cholesky factorization of a Hermitian positive definite band matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPBSTF computes a split Cholesky factorization of a complex
 * Hermitian positive definite band matrix A.
 *
 * This routine is designed to be used in conjunction with CHBGST.
 *
 * The factorization has the form  A = S**H*S  where S is a band matrix
 * of the same bandwidth as A.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     On entry, the banded matrix A.
 *                       On exit, the factor S from A = S**H*S.
 *                       Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the factorization could not be completed,
 *                           because the updated element a(i,i) was negative; the
 *                           matrix A is not positive definite.
 */
void cpbstf(
    const char* uplo,
    const INT n,
    const INT kd,
    c64* restrict AB,
    const INT ldab,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT upper;
    INT j, kld, km, m;
    f32 ajj;

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
        xerbla("CPBSTF", -(*info));
        return;
    }

    if (n == 0)
        return;

    kld = (1 > ldab - 1) ? 1 : (ldab - 1);

    // Set the splitting point m
    m = (n + kd) / 2;

    if (upper) {
        // Factorize A(m+1:n,m+1:n) as L**H*L, and update A(1:m,1:m)
        for (j = n - 1; j >= m; j--) {
            // Compute s(j,j) and test for non-positive-definiteness
            ajj = crealf(AB[kd + j * ldab]);
            if (ajj <= ZERO) {
                AB[kd + j * ldab] = CMPLXF(ajj, 0.0f);
                goto L50;
            }
            ajj = sqrtf(ajj);
            AB[kd + j * ldab] = CMPLXF(ajj, 0.0f);
            km = (j < kd) ? j : kd;

            // Compute elements j-km:j-1 of the j-th column and update
            // the leading submatrix within the band
            if (km > 0) {
                cblas_csscal(km, ONE / ajj, &AB[kd - km + j * ldab], 1);
                cblas_cher(CblasColMajor, CblasUpper, km, -ONE,
                           &AB[kd - km + j * ldab], 1,
                           &AB[kd + (j - km) * ldab], kld);
            }
        }

        // Factorize the updated submatrix A(1:m,1:m) as U**H*U
        for (j = 0; j < m; j++) {
            // Compute s(j,j) and test for non-positive-definiteness
            ajj = crealf(AB[kd + j * ldab]);
            if (ajj <= ZERO) {
                AB[kd + j * ldab] = CMPLXF(ajj, 0.0f);
                goto L50;
            }
            ajj = sqrtf(ajj);
            AB[kd + j * ldab] = CMPLXF(ajj, 0.0f);
            km = (kd < m - j - 1) ? kd : (m - j - 1);

            // Compute elements j+1:j+km of the j-th row and update
            // the trailing submatrix within the band
            if (km > 0) {
                cblas_csscal(km, ONE / ajj, &AB[kd - 1 + (j + 1) * ldab], kld);
                clacgv(km, &AB[kd - 1 + (j + 1) * ldab], kld);
                cblas_cher(CblasColMajor, CblasUpper, km, -ONE,
                           &AB[kd - 1 + (j + 1) * ldab], kld,
                           &AB[kd + (j + 1) * ldab], kld);
                clacgv(km, &AB[kd - 1 + (j + 1) * ldab], kld);
            }
        }
    } else {
        // Factorize A(m+1:n,m+1:n) as L**H*L, and update A(1:m,1:m)
        for (j = n - 1; j >= m; j--) {
            // Compute s(j,j) and test for non-positive-definiteness
            ajj = crealf(AB[0 + j * ldab]);
            if (ajj <= ZERO) {
                AB[0 + j * ldab] = CMPLXF(ajj, 0.0f);
                goto L50;
            }
            ajj = sqrtf(ajj);
            AB[0 + j * ldab] = CMPLXF(ajj, 0.0f);
            km = (j < kd) ? j : kd;

            // Compute elements j-km:j-1 of the j-th row and update
            // the trailing submatrix within the band
            if (km > 0) {
                cblas_csscal(km, ONE / ajj, &AB[km + (j - km) * ldab], kld);
                clacgv(km, &AB[km + (j - km) * ldab], kld);
                cblas_cher(CblasColMajor, CblasLower, km, -ONE,
                           &AB[km + (j - km) * ldab], kld,
                           &AB[0 + (j - km) * ldab], kld);
                clacgv(km, &AB[km + (j - km) * ldab], kld);
            }
        }

        // Factorize the updated submatrix A(1:m,1:m) as U**H*U
        for (j = 0; j < m; j++) {
            // Compute s(j,j) and test for non-positive-definiteness
            ajj = crealf(AB[0 + j * ldab]);
            if (ajj <= ZERO) {
                AB[0 + j * ldab] = CMPLXF(ajj, 0.0f);
                goto L50;
            }
            ajj = sqrtf(ajj);
            AB[0 + j * ldab] = CMPLXF(ajj, 0.0f);
            km = (kd < m - j - 1) ? kd : (m - j - 1);

            // Compute elements j+1:j+km of the j-th column and update
            // the trailing submatrix within the band
            if (km > 0) {
                cblas_csscal(km, ONE / ajj, &AB[1 + j * ldab], 1);
                cblas_cher(CblasColMajor, CblasLower, km, -ONE,
                           &AB[1 + j * ldab], 1,
                           &AB[0 + (j + 1) * ldab], kld);
            }
        }
    }
    return;

L50:
    *info = j + 1;
    return;
}
