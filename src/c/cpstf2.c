/**
 * @file cpstf2.c
 * @brief CPSTF2 computes the Cholesky factorization with complete pivoting
 *        of a complex Hermitian positive semidefinite matrix (unblocked).
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPSTF2 computes the Cholesky factorization with complete
 * pivoting of a complex Hermitian positive semidefinite matrix A.
 *
 * The factorization has the form
 *    P**T * A * P = U**H * U ,  if UPLO = 'U',
 *    P**T * A * P = L  * L**H,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular, and
 * P is stored as vector PIV.
 *
 * This algorithm does not attempt to check that A is positive
 * semidefinite. This version of the algorithm calls level 2 BLAS.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the Hermitian matrix A is stored.
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A.
 *                      On exit, if info = 0, the factor U or L from the
 *                      Cholesky factorization.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[out]    piv   Integer array, dimension (n).
 *                      PIV is such that the nonzero entries are P(PIV(k), k) = 1.
 *                      0-based indexing.
 * @param[out]    rank  The rank of A given by the number of steps the algorithm
 *                      completed.
 * @param[in]     tol   User defined tolerance. If tol < 0, then n*eps*max(A(k,k))
 *                      will be used. The algorithm terminates at the (k-1)st step
 *                      if the pivot <= tol.
 * @param[out]    work  Single precision array, dimension (2*n). Work space.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: the matrix A is either rank deficient with computed rank
 *                           as returned in rank, or is not positive semidefinite.
 */
void cpstf2(
    const char* uplo,
    const int n,
    c64* restrict A,
    const int lda,
    int* restrict piv,
    int* rank,
    const f32 tol,
    f32* restrict work,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CPSTF2", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    // Initialize PIV (0-based)
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // Compute stopping value
    int pvt = 0;
    for (int i = 0; i < n; i++) {
        work[i] = crealf(A[i + i * lda]);
    }
    f32 ajj = work[0];
    for (int i = 1; i < n; i++) {
        if (work[i] > ajj) {
            pvt = i;
            ajj = work[i];
        }
    }
    if (ajj <= ZERO || sisnan(ajj)) {
        *rank = 0;
        *info = 1;
        return;
    }

    // Compute stopping value if not supplied
    f32 dstop;
    if (tol < ZERO) {
        dstop = n * slamch("Epsilon") * ajj;
    } else {
        dstop = tol;
    }

    // Set first half of WORK to zero, holds dot products
    for (int i = 0; i < n; i++) {
        work[i] = ZERO;
    }

    int jstop = -1;

    if (upper) {
        // Compute the Cholesky factorization P**T * A * P = U**H * U
        for (int j = 0; j < n && jstop < 0; j++) {

            // Find pivot, test for exit, else swap rows and columns
            // Update dot products, compute possible pivots which are
            // stored in the second half of WORK
            for (int i = j; i < n; i++) {
                if (j > 0) {
                    work[i] = work[i] +
                              crealf(conjf(A[(j - 1) + i * lda]) *
                                    A[(j - 1) + i * lda]);
                }
                work[n + i] = crealf(A[i + i * lda]) - work[i];
            }

            if (j > 0) {
                int itemp = 0;
                f32 wmax = work[n + j];
                for (int i = 1; i < n - j; i++) {
                    if (work[n + j + i] > wmax) {
                        wmax = work[n + j + i];
                        itemp = i;
                    }
                }
                pvt = itemp + j;
                ajj = work[n + pvt];
                if (ajj <= dstop || sisnan(ajj)) {
                    A[j + j * lda] = CMPLXF(ajj, 0.0f);
                    jstop = j;
                    break;
                }
            }

            if (j != pvt) {
                // Pivot OK, so can now swap pivot rows and columns
                A[pvt + pvt * lda] = A[j + j * lda];
                if (j > 0) {
                    cblas_cswap(j, &A[j * lda], 1, &A[pvt * lda], 1);
                }
                if (pvt < n - 1) {
                    cblas_cswap(n - pvt - 1, &A[j + (pvt + 1) * lda], lda,
                                &A[pvt + (pvt + 1) * lda], lda);
                }
                for (int i = j + 1; i <= pvt - 1; i++) {
                    c64 ztemp = conjf(A[j + i * lda]);
                    A[j + i * lda] = conjf(A[i + pvt * lda]);
                    A[i + pvt * lda] = ztemp;
                }
                A[j + pvt * lda] = conjf(A[j + pvt * lda]);

                // Swap dot products and PIV
                f32 dtemp = work[j];
                work[j] = work[pvt];
                work[pvt] = dtemp;
                int itemp = piv[pvt];
                piv[pvt] = piv[j];
                piv[j] = itemp;
            }

            ajj = sqrtf(ajj);
            A[j + j * lda] = CMPLXF(ajj, 0.0f);

            // Compute elements j+1:n-1 of row j
            if (j < n - 1) {
                clacgv(j, &A[j * lda], 1);
                cblas_cgemv(CblasColMajor, CblasTrans,
                            j, n - j - 1, &NEG_CONE,
                            &A[(j + 1) * lda], lda,
                            &A[j * lda], 1,
                            &CONE, &A[j + (j + 1) * lda], lda);
                clacgv(j, &A[j * lda], 1);
                cblas_csscal(n - j - 1, ONE / ajj,
                             &A[j + (j + 1) * lda], lda);
            }
        }
    } else {
        // Compute the Cholesky factorization P**T * A * P = L * L**H
        for (int j = 0; j < n && jstop < 0; j++) {

            // Find pivot, test for exit, else swap rows and columns
            // Update dot products, compute possible pivots which are
            // stored in the second half of WORK
            for (int i = j; i < n; i++) {
                if (j > 0) {
                    work[i] = work[i] +
                              crealf(conjf(A[i + (j - 1) * lda]) *
                                    A[i + (j - 1) * lda]);
                }
                work[n + i] = crealf(A[i + i * lda]) - work[i];
            }

            if (j > 0) {
                int itemp = 0;
                f32 wmax = work[n + j];
                for (int i = 1; i < n - j; i++) {
                    if (work[n + j + i] > wmax) {
                        wmax = work[n + j + i];
                        itemp = i;
                    }
                }
                pvt = itemp + j;
                ajj = work[n + pvt];
                if (ajj <= dstop || sisnan(ajj)) {
                    A[j + j * lda] = CMPLXF(ajj, 0.0f);
                    jstop = j;
                    break;
                }
            }

            if (j != pvt) {
                // Pivot OK, so can now swap pivot rows and columns
                A[pvt + pvt * lda] = A[j + j * lda];
                if (j > 0) {
                    cblas_cswap(j, &A[j], lda, &A[pvt], lda);
                }
                if (pvt < n - 1) {
                    cblas_cswap(n - pvt - 1, &A[(pvt + 1) + j * lda], 1,
                                &A[(pvt + 1) + pvt * lda], 1);
                }
                for (int i = j + 1; i <= pvt - 1; i++) {
                    c64 ztemp = conjf(A[i + j * lda]);
                    A[i + j * lda] = conjf(A[pvt + i * lda]);
                    A[pvt + i * lda] = ztemp;
                }
                A[pvt + j * lda] = conjf(A[pvt + j * lda]);

                // Swap dot products and PIV
                f32 dtemp = work[j];
                work[j] = work[pvt];
                work[pvt] = dtemp;
                int itemp = piv[pvt];
                piv[pvt] = piv[j];
                piv[j] = itemp;
            }

            ajj = sqrtf(ajj);
            A[j + j * lda] = CMPLXF(ajj, 0.0f);

            // Compute elements j+1:n-1 of column j
            if (j < n - 1) {
                clacgv(j, &A[j], lda);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            n - j - 1, j, &NEG_CONE,
                            &A[j + 1], lda,
                            &A[j], lda,
                            &CONE, &A[(j + 1) + j * lda], 1);
                clacgv(j, &A[j], lda);
                cblas_csscal(n - j - 1, ONE / ajj,
                             &A[(j + 1) + j * lda], 1);
            }
        }
    }

    if (jstop >= 0) {
        // Rank is number of steps completed. Set info = 1 to signal
        // that the factorization cannot be used to solve a system.
        *rank = jstop;
        *info = 1;
    } else {
        // Ran to completion, A has full rank
        *rank = n;
    }
}
