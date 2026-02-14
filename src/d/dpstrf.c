/**
 * @file dpstrf.c
 * @brief DPSTRF computes the Cholesky factorization with complete pivoting
 *        of a real symmetric positive semidefinite matrix (blocked).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DPSTRF computes the Cholesky factorization with complete
 * pivoting of a real symmetric positive semidefinite matrix A.
 *
 * The factorization has the form
 *    P**T * A * P = U**T * U ,  if UPLO = 'U',
 *    P**T * A * P = L  * L**T,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular, and
 * P is stored as vector PIV.
 *
 * This algorithm does not attempt to check that A is positive
 * semidefinite. This version of the algorithm calls level 3 BLAS.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the symmetric matrix A is stored.
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the symmetric matrix A.
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
 * @param[out]    work  Double precision array, dimension (2*n). Work space.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: the matrix A is either rank deficient with computed rank
 *                           as returned in rank, or is not positive semidefinite.
 */
void dpstrf(
    const char* uplo,
    const int n,
    f64* restrict A,
    const int lda,
    int* restrict piv,
    int* rank,
    const f64 tol,
    f64* restrict work,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    const f64 NEG_ONE = -1.0;

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
        xerbla("DPSTRF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    // Get block size (uses POTRF's block size as in the original)
    int nb = lapack_get_nb("POTRF");

    if (nb <= 1 || nb >= n) {
        // Use unblocked code
        dpstf2(uplo, n, A, lda, piv, rank, tol, work, info);
        return;
    }

    // Initialize PIV (0-based)
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // Compute stopping value
    int pvt = 0;
    f64 ajj = A[0];
    for (int i = 1; i < n; i++) {
        if (A[i + i * lda] > ajj) {
            pvt = i;
            ajj = A[pvt + pvt * lda];
        }
    }
    if (ajj <= ZERO || disnan(ajj)) {
        *rank = 0;
        *info = 1;
        return;
    }

    // Compute stopping value if not supplied
    f64 dstop;
    if (tol < ZERO) {
        dstop = n * dlamch("Epsilon") * ajj;
    } else {
        dstop = tol;
    }

    int jstop = -1;  // Track where we stop due to rank deficiency

    if (upper) {
        // Compute the Cholesky factorization P**T * A * P = U**T * U
        for (int k = 0; k < n && jstop < 0; k += nb) {
            // Account for last block not being NB wide
            int jb = (nb < n - k) ? nb : (n - k);

            // Set relevant part of first half of WORK to zero, holds dot products
            for (int i = k; i < n; i++) {
                work[i] = ZERO;
            }

            for (int j = k; j < k + jb && jstop < 0; j++) {
                // Find pivot, test for exit, else swap rows and columns
                // Update dot products, compute possible pivots which are
                // stored in the second half of WORK
                for (int i = j; i < n; i++) {
                    if (j > k) {
                        f64 tmp = A[(j - 1) + i * lda];
                        work[i] = work[i] + tmp * tmp;
                    }
                    work[n + i] = A[i + i * lda] - work[i];
                }

                if (j > 0) {
                    // Find max in work[n+j : n+n-1]
                    int itemp = 0;
                    f64 wmax = work[n + j];
                    for (int i = 1; i < n - j; i++) {
                        if (work[n + j + i] > wmax) {
                            wmax = work[n + j + i];
                            itemp = i;
                        }
                    }
                    pvt = itemp + j;
                    ajj = work[n + pvt];
                    if (ajj <= dstop || disnan(ajj)) {
                        A[j + j * lda] = ajj;
                        jstop = j;
                        break;
                    }
                }

                if (j != pvt) {
                    // Pivot OK, so can now swap pivot rows and columns
                    A[pvt + pvt * lda] = A[j + j * lda];
                    if (j > 0) {
                        cblas_dswap(j, &A[j * lda], 1, &A[pvt * lda], 1);
                    }
                    if (pvt < n - 1) {
                        cblas_dswap(n - pvt - 1, &A[j + (pvt + 1) * lda], lda,
                                    &A[pvt + (pvt + 1) * lda], lda);
                    }
                    if (pvt > j + 1) {
                        cblas_dswap(pvt - j - 1, &A[j + (j + 1) * lda], lda,
                                    &A[(j + 1) + pvt * lda], 1);
                    }

                    // Swap dot products and PIV
                    f64 dtemp = work[j];
                    work[j] = work[pvt];
                    work[pvt] = dtemp;
                    int itemp = piv[pvt];
                    piv[pvt] = piv[j];
                    piv[j] = itemp;
                }

                ajj = sqrt(ajj);
                A[j + j * lda] = ajj;

                // Compute elements j+1:n-1 of row j
                if (j < n - 1) {
                    // DGEMV('Trans', j-k, n-j-1, -ONE, A(k,j+1), lda, A(k,j), 1, ONE, A(j,j+1), lda)
                    if (j > k) {
                        cblas_dgemv(CblasColMajor, CblasTrans, j - k, n - j - 1,
                                    NEG_ONE, &A[k + (j + 1) * lda], lda,
                                    &A[k + j * lda], 1,
                                    ONE, &A[j + (j + 1) * lda], lda);
                    }
                    cblas_dscal(n - j - 1, ONE / ajj, &A[j + (j + 1) * lda], lda);
                }
            }

            // Update trailing matrix, j already incremented
            // After the inner loop, if we didn't stop early, we're at j = k + jb
            if (jstop < 0 && k + jb < n) {
                // DSYRK('Upper', 'Trans', n-(k+jb), jb, -ONE, A(k,k+jb), lda, ONE, A(k+jb,k+jb), lda)
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                            n - k - jb, jb, NEG_ONE,
                            &A[k + (k + jb) * lda], lda,
                            ONE, &A[(k + jb) + (k + jb) * lda], lda);
            }
        }
    } else {
        // Compute the Cholesky factorization P**T * A * P = L * L**T
        for (int k = 0; k < n && jstop < 0; k += nb) {
            // Account for last block not being NB wide
            int jb = (nb < n - k) ? nb : (n - k);

            // Set relevant part of first half of WORK to zero, holds dot products
            for (int i = k; i < n; i++) {
                work[i] = ZERO;
            }

            for (int j = k; j < k + jb && jstop < 0; j++) {
                // Find pivot, test for exit, else swap rows and columns
                // Update dot products, compute possible pivots which are
                // stored in the second half of WORK
                for (int i = j; i < n; i++) {
                    if (j > k) {
                        f64 tmp = A[i + (j - 1) * lda];
                        work[i] = work[i] + tmp * tmp;
                    }
                    work[n + i] = A[i + i * lda] - work[i];
                }

                if (j > 0) {
                    // Find max in work[n+j : n+n-1]
                    int itemp = 0;
                    f64 wmax = work[n + j];
                    for (int i = 1; i < n - j; i++) {
                        if (work[n + j + i] > wmax) {
                            wmax = work[n + j + i];
                            itemp = i;
                        }
                    }
                    pvt = itemp + j;
                    ajj = work[n + pvt];
                    if (ajj <= dstop || disnan(ajj)) {
                        A[j + j * lda] = ajj;
                        jstop = j;
                        break;
                    }
                }

                if (j != pvt) {
                    // Pivot OK, so can now swap pivot rows and columns
                    A[pvt + pvt * lda] = A[j + j * lda];
                    if (j > 0) {
                        cblas_dswap(j, &A[j], lda, &A[pvt], lda);
                    }
                    if (pvt < n - 1) {
                        cblas_dswap(n - pvt - 1, &A[(pvt + 1) + j * lda], 1,
                                    &A[(pvt + 1) + pvt * lda], 1);
                    }
                    if (pvt > j + 1) {
                        cblas_dswap(pvt - j - 1, &A[(j + 1) + j * lda], 1,
                                    &A[pvt + (j + 1) * lda], lda);
                    }

                    // Swap dot products and PIV
                    f64 dtemp = work[j];
                    work[j] = work[pvt];
                    work[pvt] = dtemp;
                    int itemp = piv[pvt];
                    piv[pvt] = piv[j];
                    piv[j] = itemp;
                }

                ajj = sqrt(ajj);
                A[j + j * lda] = ajj;

                // Compute elements j+1:n-1 of column j
                if (j < n - 1) {
                    // DGEMV('No Trans', n-j-1, j-k, -ONE, A(j+1,k), lda, A(j,k), lda, ONE, A(j+1,j), 1)
                    if (j > k) {
                        cblas_dgemv(CblasColMajor, CblasNoTrans, n - j - 1, j - k,
                                    NEG_ONE, &A[(j + 1) + k * lda], lda,
                                    &A[j + k * lda], lda,
                                    ONE, &A[(j + 1) + j * lda], 1);
                    }
                    cblas_dscal(n - j - 1, ONE / ajj, &A[(j + 1) + j * lda], 1);
                }
            }

            // Update trailing matrix, j already incremented
            if (jstop < 0 && k + jb < n) {
                // DSYRK('Lower', 'No Trans', n-(k+jb), jb, -ONE, A(k+jb,k), lda, ONE, A(k+jb,k+jb), lda)
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            n - k - jb, jb, NEG_ONE,
                            &A[(k + jb) + k * lda], lda,
                            ONE, &A[(k + jb) + (k + jb) * lda], lda);
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
