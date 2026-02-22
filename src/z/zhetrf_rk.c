/**
 * @file zhetrf_rk.c
 * @brief ZHETRF_RK computes the factorization of a complex Hermitian indefinite matrix using the bounded Bunch-Kaufman (rook) diagonal pivoting method (BLAS3 blocked algorithm).
 */

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZHETRF_RK computes the factorization of a complex Hermitian matrix A
 * using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
 *
 *    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**H (or L**H) is the conjugate of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is Hermitian and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of the
 *          Hermitian matrix A is stored:
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the Hermitian matrix A.
 *          On exit, contains:
 *            a) ONLY diagonal elements of the Hermitian block diagonal
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
 *          Double complex array, dimension (n).
 *          On exit, contains the superdiagonal (or subdiagonal)
 *          elements of the Hermitian block diagonal matrix D
 *          with 1-by-1 or 2-by-2 diagonal blocks, where
 *          If UPLO = 'U': E(i) = D(i-1,i), i=2:N, E(1) is set to 0;
 *          If UPLO = 'L': E(i) = D(i+1,i), i=1:N-1, E(N) is set to 0.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          IPIV describes the permutation matrix P in the factorization.
 *
 * @param[out] work
 *          Double complex array, dimension (max(1, lwork)).
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
void zhetrf_rk(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    c128* restrict E,
    INT* restrict ipiv,
    c128* restrict work,
    const INT lwork,
    INT* info)
{
    INT upper, lquery;
    INT i, iinfo, ip, iws, k, kb, ldwork, lwkopt, nb, nbmin;

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
        nb = lapack_get_nb("HETRF");
        lwkopt = (1 > n * nb) ? 1 : n * nb;
        work[0] = (c128)lwkopt;
    }

    if (*info != 0) {
        xerbla("ZHETRF_RK", -(*info));
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
            nbmin = lapack_get_nbmin("HETRF");
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

                zlahef_rk(uplo, k + 1, nb, &kb, A, lda, E, ipiv, work, ldwork, &iinfo);

            } else {

                zhetf2_rk(uplo, k + 1, A, lda, E, ipiv, &iinfo);
                kb = k + 1;
            }

            if (*info == 0 && iinfo > 0) {
                *info = iinfo;
            }

            if (k < n - 1) {
                for (i = k; i >= k - kb + 1; i--) {
                    ip = (ipiv[i] >= 0) ? ipiv[i] : -(ipiv[i] + 1);
                    if (ip != i) {
                        cblas_zswap(n - k - 1, &A[i + (k + 1) * lda], lda, &A[ip + (k + 1) * lda], lda);
                    }
                }
            }

            k = k - kb;
        }

    } else {

        k = 0;

        while (k < n) {

            if (k <= n - 1 - nb) {

                zlahef_rk(uplo, n - k, nb, &kb, &A[k + k * lda], lda, &E[k],
                          &ipiv[k], work, ldwork, &iinfo);

            } else {

                zhetf2_rk(uplo, n - k, &A[k + k * lda], lda, &E[k], &ipiv[k], &iinfo);
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
                        cblas_zswap(k, &A[i + 0 * lda], lda, &A[ip + 0 * lda], lda);
                    }
                }
            }

            k = k + kb;
        }
    }

    work[0] = (c128)lwkopt;
}
