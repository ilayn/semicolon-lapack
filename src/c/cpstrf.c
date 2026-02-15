/**
 * @file cpstrf.c
 * @brief CPSTRF computes the Cholesky factorization with complete pivoting
 *        of a complex Hermitian positive semidefinite matrix (blocked).
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"

/**
 * CPSTRF computes the Cholesky factorization with complete
 * pivoting of a complex Hermitian positive semidefinite matrix A.
 *
 * The factorization has the form
 *    P**T * A * P = U**H * U ,  if UPLO = 'U',
 *    P**T * A * P = L  * L**H,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular, and
 * P is stored as vector PIV.
 *
 * This algorithm does not attempt to check that A is positive
 * semidefinite. This version of the algorithm calls level 3 BLAS.
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
void cpstrf(
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
    const f32 NEG_ONE = -1.0f;
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
        xerbla("CPSTRF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    int nb = lapack_get_nb("POTRF");

    if (nb <= 1 || nb >= n) {
        cpstf2(uplo, n, A, lda, piv, rank, tol, work, info);
        return;
    }

    // Initialize PIV (0-based)
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // Compute stopping value
    int pvt = 0;
    f32 ajj = crealf(A[0]);
    for (int i = 1; i < n; i++) {
        f32 tmp = crealf(A[i + i * lda]);
        if (tmp > ajj) {
            pvt = i;
            ajj = tmp;
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

    int jstop = -1;

    if (upper) {
        // Compute the Cholesky factorization P**T * A * P = U**H * U
        for (int k = 0; k < n && jstop < 0; k += nb) {
            int jb = (nb < n - k) ? nb : (n - k);

            for (int i = k; i < n; i++) {
                work[i] = ZERO;
            }

            for (int j = k; j < k + jb && jstop < 0; j++) {
                for (int i = j; i < n; i++) {
                    if (j > k) {
                        c64 val = A[(j - 1) + i * lda];
                        work[i] = work[i] + crealf(conjf(val) * val);
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
                    A[pvt + pvt * lda] = A[j + j * lda];
                    if (j > 0) {
                        cblas_cswap(j, &A[j * lda], 1, &A[pvt * lda], 1);
                    }
                    if (pvt < n - 1) {
                        cblas_cswap(n - pvt - 1, &A[j + (pvt + 1) * lda], lda,
                                    &A[pvt + (pvt + 1) * lda], lda);
                    }
                    for (int i = j + 1; i < pvt; i++) {
                        c64 ztemp = conjf(A[j + i * lda]);
                        A[j + i * lda] = conjf(A[i + pvt * lda]);
                        A[i + pvt * lda] = ztemp;
                    }
                    A[j + pvt * lda] = conjf(A[j + pvt * lda]);

                    f32 dtemp = work[j];
                    work[j] = work[pvt];
                    work[pvt] = dtemp;
                    int itemp = piv[pvt];
                    piv[pvt] = piv[j];
                    piv[j] = itemp;
                }

                ajj = sqrtf(ajj);
                A[j + j * lda] = CMPLXF(ajj, 0.0f);

                if (j < n - 1) {
                    clacgv(j, &A[j * lda], 1);
                    if (j > k) {
                        cblas_cgemv(CblasColMajor, CblasTrans, j - k, n - j - 1,
                                    &NEG_CONE, &A[k + (j + 1) * lda], lda,
                                    &A[k + j * lda], 1,
                                    &CONE, &A[j + (j + 1) * lda], lda);
                    }
                    clacgv(j, &A[j * lda], 1);
                    cblas_csscal(n - j - 1, ONE / ajj, &A[j + (j + 1) * lda], lda);
                }
            }

            if (jstop < 0 && k + jb < n) {
                cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                            n - k - jb, jb, NEG_ONE,
                            &A[k + (k + jb) * lda], lda,
                            ONE, &A[(k + jb) + (k + jb) * lda], lda);
            }
        }
    } else {
        // Compute the Cholesky factorization P**T * A * P = L * L**H
        for (int k = 0; k < n && jstop < 0; k += nb) {
            int jb = (nb < n - k) ? nb : (n - k);

            for (int i = k; i < n; i++) {
                work[i] = ZERO;
            }

            for (int j = k; j < k + jb && jstop < 0; j++) {
                for (int i = j; i < n; i++) {
                    if (j > k) {
                        c64 val = A[i + (j - 1) * lda];
                        work[i] = work[i] + crealf(conjf(val) * val);
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
                    A[pvt + pvt * lda] = A[j + j * lda];
                    if (j > 0) {
                        cblas_cswap(j, &A[j], lda, &A[pvt], lda);
                    }
                    if (pvt < n - 1) {
                        cblas_cswap(n - pvt - 1, &A[(pvt + 1) + j * lda], 1,
                                    &A[(pvt + 1) + pvt * lda], 1);
                    }
                    for (int i = j + 1; i < pvt; i++) {
                        c64 ztemp = conjf(A[i + j * lda]);
                        A[i + j * lda] = conjf(A[pvt + i * lda]);
                        A[pvt + i * lda] = ztemp;
                    }
                    A[pvt + j * lda] = conjf(A[pvt + j * lda]);

                    f32 dtemp = work[j];
                    work[j] = work[pvt];
                    work[pvt] = dtemp;
                    int itemp = piv[pvt];
                    piv[pvt] = piv[j];
                    piv[j] = itemp;
                }

                ajj = sqrtf(ajj);
                A[j + j * lda] = CMPLXF(ajj, 0.0f);

                if (j < n - 1) {
                    clacgv(j, &A[j], lda);
                    if (j > k) {
                        cblas_cgemv(CblasColMajor, CblasNoTrans, n - j - 1, j - k,
                                    &NEG_CONE, &A[(j + 1) + k * lda], lda,
                                    &A[j + k * lda], lda,
                                    &CONE, &A[(j + 1) + j * lda], 1);
                    }
                    clacgv(j, &A[j], lda);
                    cblas_csscal(n - j - 1, ONE / ajj, &A[(j + 1) + j * lda], 1);
                }
            }

            if (jstop < 0 && k + jb < n) {
                cblas_cherk(CblasColMajor, CblasLower, CblasNoTrans,
                            n - k - jb, jb, NEG_ONE,
                            &A[(k + jb) + k * lda], lda,
                            ONE, &A[(k + jb) + (k + jb) * lda], lda);
            }
        }
    }

    if (jstop >= 0) {
        *rank = jstop;
        *info = 1;
    } else {
        *rank = n;
    }
}
