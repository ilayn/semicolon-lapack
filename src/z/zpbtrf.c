/**
 * @file zpbtrf.c
 * @brief ZPBTRF computes the Cholesky factorization of a Hermitian positive definite band matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

#define NBMAX 32
#define LDWORK (NBMAX + 1)

/**
 * ZPBTRF computes the Cholesky factorization of a complex Hermitian
 * positive definite band matrix A.
 *
 * The factorization has the form
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L * L**H,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, the factor U or L.
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = k, the leading minor of order k is not
 *                           positive definite.
 */
void zpbtrf(
    const char* uplo,
    const int n,
    const int kd,
    c128* const restrict AB,
    const int ldab,
    int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;

    int i, i2, i3, ib, ii, j, jj, nb;
    c128 work[LDWORK * NBMAX];
    int upper;

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
        xerbla("ZPBTRF", -(*info));
        return;
    }

    if (n == 0)
        return;

    // Default block size from ilaenv is 64, but limited to NBMAX=32
    nb = 64;
    if (nb > NBMAX)
        nb = NBMAX;

    if (nb <= 1 || nb > kd) {
        // Use unblocked code
        zpbtf2(uplo, n, kd, AB, ldab, info);
    } else {
        // Use blocked code
        if (upper) {
            // Zero the upper triangle of the work array
            for (j = 0; j < nb; j++) {
                for (i = 0; i < j; i++) {
                    work[i + j * LDWORK] = CZERO;
                }
            }

            // Process the band matrix one diagonal block at a time
            for (i = 0; i < n; i += nb) {
                ib = (nb < n - i) ? nb : (n - i);

                // Factorize the diagonal block
                zpotf2(uplo, ib, &AB[kd + i * ldab], ldab - 1, &ii);
                if (ii != 0) {
                    *info = i + ii;
                    return;
                }
                if (i + ib < n) {
                    // Update the relevant part of the trailing submatrix
                    i2 = (kd - ib < n - i - ib) ? (kd - ib) : (n - i - ib);
                    i3 = (ib < n - i - kd) ? ib : (n - i - kd);
                    if (i3 < 0) i3 = 0;

                    if (i2 > 0) {
                        // Update A12
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                                    CblasNonUnit, ib, i2, &CONE, &AB[kd + i * ldab],
                                    ldab - 1, &AB[kd - ib + (i + ib) * ldab], ldab - 1);
                        // Update A22
                        cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, i2, ib, NEG_ONE,
                                    &AB[kd - ib + (i + ib) * ldab], ldab - 1, ONE,
                                    &AB[kd + (i + ib) * ldab], ldab - 1);
                    }

                    if (i3 > 0) {
                        // Copy the lower triangle of A13 into the work array
                        for (jj = 0; jj < i3; jj++) {
                            for (ii = jj; ii < ib; ii++) {
                                work[ii + jj * LDWORK] = AB[ii - jj + (jj + i + kd) * ldab];
                            }
                        }
                        // Update A13 (in the work array)
                        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                                    CblasNonUnit, ib, i3, &CONE, &AB[kd + i * ldab],
                                    ldab - 1, work, LDWORK);
                        // Update A23
                        if (i2 > 0) {
                            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, i2, i3,
                                        ib, &NEG_CONE, &AB[kd - ib + (i + ib) * ldab],
                                        ldab - 1, work, LDWORK, &CONE,
                                        &AB[ib + (i + kd) * ldab], ldab - 1);
                        }
                        // Update A33
                        cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, i3, ib, NEG_ONE,
                                    work, LDWORK, ONE, &AB[kd + (i + kd) * ldab],
                                    ldab - 1);
                        // Copy the lower triangle of A13 back into place
                        for (jj = 0; jj < i3; jj++) {
                            for (ii = jj; ii < ib; ii++) {
                                AB[ii - jj + (jj + i + kd) * ldab] = work[ii + jj * LDWORK];
                            }
                        }
                    }
                }
            }
        } else {
            // Zero the lower triangle of the work array
            for (j = 0; j < nb; j++) {
                for (i = j + 1; i < nb; i++) {
                    work[i + j * LDWORK] = CZERO;
                }
            }

            // Process the band matrix one diagonal block at a time
            for (i = 0; i < n; i += nb) {
                ib = (nb < n - i) ? nb : (n - i);

                // Factorize the diagonal block
                zpotf2(uplo, ib, &AB[0 + i * ldab], ldab - 1, &ii);
                if (ii != 0) {
                    *info = i + ii;
                    return;
                }
                if (i + ib < n) {
                    // Update the relevant part of the trailing submatrix
                    i2 = (kd - ib < n - i - ib) ? (kd - ib) : (n - i - ib);
                    i3 = (ib < n - i - kd) ? ib : (n - i - kd);
                    if (i3 < 0) i3 = 0;

                    if (i2 > 0) {
                        // Update A21
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                                    CblasNonUnit, i2, ib, &CONE, &AB[0 + i * ldab],
                                    ldab - 1, &AB[ib + i * ldab], ldab - 1);
                        // Update A22
                        cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, i2, ib, NEG_ONE,
                                    &AB[ib + i * ldab], ldab - 1, ONE,
                                    &AB[0 + (i + ib) * ldab], ldab - 1);
                    }

                    if (i3 > 0) {
                        // Copy the upper triangle of A31 into the work array
                        for (jj = 0; jj < ib; jj++) {
                            int minval = (jj + 1 < i3) ? (jj + 1) : i3;
                            for (ii = 0; ii < minval; ii++) {
                                work[ii + jj * LDWORK] = AB[kd - jj + ii + (jj + i) * ldab];
                            }
                        }
                        // Update A31 (in the work array)
                        cblas_ztrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                                    CblasNonUnit, i3, ib, &CONE, &AB[0 + i * ldab],
                                    ldab - 1, work, LDWORK);
                        // Update A32
                        if (i2 > 0) {
                            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, i3, i2,
                                        ib, &NEG_CONE, work, LDWORK,
                                        &AB[ib + i * ldab], ldab - 1, &CONE,
                                        &AB[kd - ib + (i + ib) * ldab], ldab - 1);
                        }
                        // Update A33
                        cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans, i3, ib, NEG_ONE,
                                    work, LDWORK, ONE, &AB[0 + (i + kd) * ldab],
                                    ldab - 1);
                        // Copy the upper triangle of A31 back into place
                        for (jj = 0; jj < ib; jj++) {
                            int minval = (jj + 1 < i3) ? (jj + 1) : i3;
                            for (ii = 0; ii < minval; ii++) {
                                AB[kd - jj + ii + (jj + i) * ldab] = work[ii + jj * LDWORK];
                            }
                        }
                    }
                }
            }
        }
    }
}
