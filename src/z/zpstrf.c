/**
 * @file zpstrf.c
 * @brief ZPSTRF computes the Cholesky factorization with complete pivoting
 *        of a complex Hermitian positive semidefinite matrix (blocked).
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZPSTRF computes the Cholesky factorization with complete
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
 * @param[out]    work  Double precision array, dimension (2*n). Work space.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: the matrix A is either rank deficient with computed rank
 *                           as returned in rank, or is not positive semidefinite.
 */
void zpstrf(
    const char* uplo,
    const int n,
    double complex* const restrict A,
    const int lda,
    int* const restrict piv,
    int* rank,
    const double tol,
    double* const restrict work,
    int* info)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;
    const double NEG_ONE = -1.0;
    const double complex CONE = CMPLX(1.0, 0.0);
    const double complex NEG_CONE = CMPLX(-1.0, 0.0);

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
        xerbla("ZPSTRF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    int nb = lapack_get_nb("POTRF");

    if (nb <= 1 || nb >= n) {
        zpstf2(uplo, n, A, lda, piv, rank, tol, work, info);
        return;
    }

    // Initialize PIV (0-based)
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // Compute stopping value
    int pvt = 0;
    double ajj = creal(A[0]);
    for (int i = 1; i < n; i++) {
        double tmp = creal(A[i + i * lda]);
        if (tmp > ajj) {
            pvt = i;
            ajj = tmp;
        }
    }
    if (ajj <= ZERO || disnan(ajj)) {
        *rank = 0;
        *info = 1;
        return;
    }

    // Compute stopping value if not supplied
    double dstop;
    if (tol < ZERO) {
        dstop = n * dlamch("Epsilon") * ajj;
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
                        double complex val = A[(j - 1) + i * lda];
                        work[i] = work[i] + creal(conj(val) * val);
                    }
                    work[n + i] = creal(A[i + i * lda]) - work[i];
                }

                if (j > 0) {
                    int itemp = 0;
                    double wmax = work[n + j];
                    for (int i = 1; i < n - j; i++) {
                        if (work[n + j + i] > wmax) {
                            wmax = work[n + j + i];
                            itemp = i;
                        }
                    }
                    pvt = itemp + j;
                    ajj = work[n + pvt];
                    if (ajj <= dstop || disnan(ajj)) {
                        A[j + j * lda] = CMPLX(ajj, 0.0);
                        jstop = j;
                        break;
                    }
                }

                if (j != pvt) {
                    A[pvt + pvt * lda] = A[j + j * lda];
                    if (j > 0) {
                        cblas_zswap(j, &A[j * lda], 1, &A[pvt * lda], 1);
                    }
                    if (pvt < n - 1) {
                        cblas_zswap(n - pvt - 1, &A[j + (pvt + 1) * lda], lda,
                                    &A[pvt + (pvt + 1) * lda], lda);
                    }
                    for (int i = j + 1; i < pvt; i++) {
                        double complex ztemp = conj(A[j + i * lda]);
                        A[j + i * lda] = conj(A[i + pvt * lda]);
                        A[i + pvt * lda] = ztemp;
                    }
                    A[j + pvt * lda] = conj(A[j + pvt * lda]);

                    double dtemp = work[j];
                    work[j] = work[pvt];
                    work[pvt] = dtemp;
                    int itemp = piv[pvt];
                    piv[pvt] = piv[j];
                    piv[j] = itemp;
                }

                ajj = sqrt(ajj);
                A[j + j * lda] = CMPLX(ajj, 0.0);

                if (j < n - 1) {
                    zlacgv(j, &A[j * lda], 1);
                    if (j > k) {
                        cblas_zgemv(CblasColMajor, CblasTrans, j - k, n - j - 1,
                                    &NEG_CONE, &A[k + (j + 1) * lda], lda,
                                    &A[k + j * lda], 1,
                                    &CONE, &A[j + (j + 1) * lda], lda);
                    }
                    zlacgv(j, &A[j * lda], 1);
                    cblas_zdscal(n - j - 1, ONE / ajj, &A[j + (j + 1) * lda], lda);
                }
            }

            if (jstop < 0 && k + jb < n) {
                cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
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
                        double complex val = A[i + (j - 1) * lda];
                        work[i] = work[i] + creal(conj(val) * val);
                    }
                    work[n + i] = creal(A[i + i * lda]) - work[i];
                }

                if (j > 0) {
                    int itemp = 0;
                    double wmax = work[n + j];
                    for (int i = 1; i < n - j; i++) {
                        if (work[n + j + i] > wmax) {
                            wmax = work[n + j + i];
                            itemp = i;
                        }
                    }
                    pvt = itemp + j;
                    ajj = work[n + pvt];
                    if (ajj <= dstop || disnan(ajj)) {
                        A[j + j * lda] = CMPLX(ajj, 0.0);
                        jstop = j;
                        break;
                    }
                }

                if (j != pvt) {
                    A[pvt + pvt * lda] = A[j + j * lda];
                    if (j > 0) {
                        cblas_zswap(j, &A[j], lda, &A[pvt], lda);
                    }
                    if (pvt < n - 1) {
                        cblas_zswap(n - pvt - 1, &A[(pvt + 1) + j * lda], 1,
                                    &A[(pvt + 1) + pvt * lda], 1);
                    }
                    for (int i = j + 1; i < pvt; i++) {
                        double complex ztemp = conj(A[i + j * lda]);
                        A[i + j * lda] = conj(A[pvt + i * lda]);
                        A[pvt + i * lda] = ztemp;
                    }
                    A[pvt + j * lda] = conj(A[pvt + j * lda]);

                    double dtemp = work[j];
                    work[j] = work[pvt];
                    work[pvt] = dtemp;
                    int itemp = piv[pvt];
                    piv[pvt] = piv[j];
                    piv[j] = itemp;
                }

                ajj = sqrt(ajj);
                A[j + j * lda] = CMPLX(ajj, 0.0);

                if (j < n - 1) {
                    zlacgv(j, &A[j], lda);
                    if (j > k) {
                        cblas_zgemv(CblasColMajor, CblasNoTrans, n - j - 1, j - k,
                                    &NEG_CONE, &A[(j + 1) + k * lda], lda,
                                    &A[j + k * lda], lda,
                                    &CONE, &A[(j + 1) + j * lda], 1);
                    }
                    zlacgv(j, &A[j], lda);
                    cblas_zdscal(n - j - 1, ONE / ajj, &A[(j + 1) + j * lda], 1);
                }
            }

            if (jstop < 0 && k + jb < n) {
                cblas_zherk(CblasColMajor, CblasLower, CblasNoTrans,
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
