/**
 * @file ssytrf_rk.c
 * @brief SSYTRF_RK computes the factorization of a real symmetric indefinite matrix using the bounded Bunch-Kaufman (rook) diagonal pivoting method (BLAS3 blocked algorithm).
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SSYTRF_RK computes the factorization of a real symmetric matrix A
 * using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
 *
 *    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of the
 *          symmetric matrix A is stored:
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, contains:
 *            a) ONLY diagonal elements of the symmetric block diagonal
 *               matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
 *               (superdiagonal (or subdiagonal) elements of D
 *                are stored on exit in array E), and
 *            b) If UPLO = 'U': factor U in the superdiagonal part of A.
 *               If UPLO = 'L': factor L in the subdiagonal part of A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] E
 *          Double precision array, dimension (n).
 *          On exit, contains the superdiagonal (or subdiagonal)
 *          elements of the symmetric block diagonal matrix D.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          IPIV describes the permutation matrix P in the factorization.
 *
 * @param[out] work
 *          Double precision array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The length of work. lwork >= 1. For best performance
 *          lwork >= n*nb, where nb is the block size.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the matrix A is singular.
 */
void ssytrf_rk(
    const char* uplo,
    const int n,
    f32* restrict A,
    const int lda,
    f32* restrict E,
    int* restrict ipiv,
    f32* restrict work,
    const int lwork,
    int* info)
{
    int upper, lquery;
    int i, iinfo, ip, iws, k, kb, ldwork, lwkopt, nb, nbmin;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (lwork < 1 && !lquery) {
        *info = -8;
    }

    if (*info == 0) {
        nb = lapack_get_nb("SYTRF");
        lwkopt = (1 > n * nb) ? 1 : n * nb;
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SSYTRF_RK", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    nbmin = 2;
    ldwork = n;
    if (nb > 1 && nb < n) {
        iws = ldwork * nb;
        if (lwork < iws) {
            nb = lwork / ldwork;
            if (nb < 1) {
                nb = 1;
            }
            nbmin = lapack_get_nbmin("SYTRF");
            if (nbmin < 2) {
                nbmin = 2;
            }
        }
    } else {
    }
    if (nb < nbmin) {
        nb = n;
    }

    if (upper) {

        k = n - 1;

        while (k >= 0) {

            if (k + 1 > nb) {

                slasyf_rk(uplo, k + 1, nb, &kb, A, lda, E, ipiv, work, ldwork, &iinfo);

            } else {

                ssytf2_rk(uplo, k + 1, A, lda, E, ipiv, &iinfo);
                kb = k + 1;
            }

            if (*info == 0 && iinfo > 0) {
                *info = iinfo;
            }

            if (k < n - 1) {
                for (i = k; i >= k - kb + 1; i--) {
                    ip = (ipiv[i] >= 0) ? ipiv[i] : -(ipiv[i] + 1);
                    if (ip != i) {
                        cblas_sswap(n - k - 1, &A[i + (k + 1) * lda], lda, &A[ip + (k + 1) * lda], lda);
                    }
                }
            }

            k = k - kb;
        }

    } else {

        k = 0;

        while (k < n) {

            if (k <= n - 1 - nb) {

                slasyf_rk(uplo, n - k, nb, &kb, &A[k + k * lda], lda, &E[k],
                          &ipiv[k], work, ldwork, &iinfo);

            } else {

                ssytf2_rk(uplo, n - k, &A[k + k * lda], lda, &E[k], &ipiv[k], &iinfo);
                kb = n - k;
            }

            if (*info == 0 && iinfo > 0) {
                *info = iinfo + k;
            }

            for (i = k; i < k + kb; i++) {
                if (ipiv[i] >= 0) {
                    ipiv[i] = ipiv[i] + k;
                } else {
                    ipiv[i] = ipiv[i] - k;
                }
            }

            if (k > 0) {
                for (i = k; i < k + kb; i++) {
                    ip = (ipiv[i] >= 0) ? ipiv[i] : -(ipiv[i] + 1);
                    if (ip != i) {
                        cblas_sswap(k, &A[i + 0 * lda], lda, &A[ip + 0 * lda], lda);
                    }
                }
            }

            k = k + kb;
        }
    }

    work[0] = (f32)lwkopt;
}
