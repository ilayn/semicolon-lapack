/**
 * @file cpbtrf.c
 * @brief CPBTRF computes the Cholesky factorization of a Hermitian positive definite band matrix.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

#define NBMAX 32
#define LDWORK (NBMAX + 1)

/**
 * CPBTRF computes the Cholesky factorization of a complex Hermitian
 * positive definite band matrix A.
 *
 * The factorization has the form
 * @rst
 * .. code-block:: text
 *
 *     A = U**H * U,  if uplo = 'U', or
 *     A = L * L**H,  if uplo = 'L',
 * @endrst
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in,out] AB     Array of dimension (`ldab`, `n`).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of A is stored in the j-th
 *                       column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 *                       On exit, if `info=0`, the triangular factor U or L
 *                       from the Cholesky factorization A = U**H*U or
 *                       A = L*L**H of the band matrix A, in the same storage
 *                       format as A.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, the leading principal minor of order
 *                           i is not positive, and the factorization could not be
 *                           completed.
 *
 * @par Further Details:
 * @rst
 * The band storage scheme is illustrated by the following example, when
 * ``n=6``, ``kd=2``, and ``uplo='U'``:
 *
 * .. code-block:: text
 *
 *     On entry:                       On exit:
 *
 *          *    *   a02  a13  a24  a35      *    *   u02  u13  u24  u35
 *          *   a01  a12  a23  a34  a45      *   u01  u12  u23  u34  u45
 *         a00  a11  a22  a33  a44  a55     u00  u11  u22  u33  u44  u55
 *
 * Similarly, if ``uplo='L'`` the format of A is as follows:
 *
 * .. code-block:: text
 *
 *     On entry:                       On exit:
 *
 *         a00  a11  a22  a33  a44  a55     l00  l11  l22  l33  l44  l55
 *         a10  a21  a32  a43  a54   *      l10  l21  l32  l43  l54   *
 *         a20  a31  a42  a53   *    *      l20  l31  l42  l53   *    *
 *
 * Array elements marked * are not used by the routine.
 * @endrst
 *
 * @par Contributors:
 * @rst
 * Peter Mayes and Giuseppe Radicati, IBM ECSEC, Rome, March 23, 1989
 * @endrst
 */
void cpbtrf(
    const char* uplo,
    const INT n,
    const INT kd,
    c64* restrict AB,
    const INT ldab,
    INT* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;

    INT i, i2, i3, ib, ii, j, jj, nb;
    c64 work[LDWORK * NBMAX];
    INT upper;

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
        xerbla("CPBTRF", -(*info));
        return;
    }

    if (n == 0)
        return;

    nb = lapack_get_nb("PBTRF");
    if (nb > NBMAX)
        nb = NBMAX;

    if (nb <= 1 || nb > kd) {
        // Use unblocked code
        cpbtf2(uplo, n, kd, AB, ldab, info);
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
                cpotf2(uplo, ib, &AB[kd + i * ldab], ldab - 1, &ii);
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
                        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                                    CblasNonUnit, ib, i2, &CONE, &AB[kd + i * ldab],
                                    ldab - 1, &AB[kd - ib + (i + ib) * ldab], ldab - 1);
                        // Update A22
                        cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans, i2, ib, NEG_ONE,
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
                        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                                    CblasNonUnit, ib, i3, &CONE, &AB[kd + i * ldab],
                                    ldab - 1, work, LDWORK);
                        // Update A23
                        if (i2 > 0) {
                            cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, i2, i3,
                                        ib, &NEG_CONE, &AB[kd - ib + (i + ib) * ldab],
                                        ldab - 1, work, LDWORK, &CONE,
                                        &AB[ib + (i + kd) * ldab], ldab - 1);
                        }
                        // Update A33
                        cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans, i3, ib, NEG_ONE,
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
                cpotf2(uplo, ib, &AB[0 + i * ldab], ldab - 1, &ii);
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
                        cblas_ctrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                                    CblasNonUnit, i2, ib, &CONE, &AB[0 + i * ldab],
                                    ldab - 1, &AB[ib + i * ldab], ldab - 1);
                        // Update A22
                        cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans, i2, ib, NEG_ONE,
                                    &AB[ib + i * ldab], ldab - 1, ONE,
                                    &AB[0 + (i + ib) * ldab], ldab - 1);
                    }

                    if (i3 > 0) {
                        // Copy the upper triangle of A31 into the work array
                        for (jj = 0; jj < ib; jj++) {
                            INT minval = (jj + 1 < i3) ? (jj + 1) : i3;
                            for (ii = 0; ii < minval; ii++) {
                                work[ii + jj * LDWORK] = AB[kd - jj + ii + (jj + i) * ldab];
                            }
                        }
                        // Update A31 (in the work array)
                        cblas_ctrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                                    CblasNonUnit, i3, ib, &CONE, &AB[0 + i * ldab],
                                    ldab - 1, work, LDWORK);
                        // Update A32
                        if (i2 > 0) {
                            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, i3, i2,
                                        ib, &NEG_CONE, work, LDWORK,
                                        &AB[ib + i * ldab], ldab - 1, &CONE,
                                        &AB[kd - ib + (i + ib) * ldab], ldab - 1);
                        }
                        // Update A33
                        cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans, i3, ib, NEG_ONE,
                                    work, LDWORK, ONE, &AB[0 + (i + kd) * ldab],
                                    ldab - 1);
                        // Copy the upper triangle of A31 back into place
                        for (jj = 0; jj < ib; jj++) {
                            INT minval = (jj + 1 < i3) ? (jj + 1) : i3;
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
