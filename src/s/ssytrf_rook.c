/**
 * @file ssytrf_rook.c
 * @brief SSYTRF_ROOK computes the factorization of a real symmetric matrix using the bounded Bunch-Kaufman ("rook") diagonal pivoting method.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SSYTRF_ROOK computes the factorization of a real symmetric matrix A
 * using the bounded Bunch-Kaufman ("rook") diagonal pivoting method.
 * The form of the factorization is
 *
 *    A = U*D*U**T  or  A = L*D*L**T
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and D is symmetric and block diagonal with
 * 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, the block diagonal matrix D and the multipliers
 *          used to obtain the factor U or L.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
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
 *                         - > 0: if info = i, D(i,i) is exactly zero. The factorization
 *                           has been completed, but the block diagonal matrix D is
 *                           exactly singular.
 */
void ssytrf_rook(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    INT* restrict ipiv,
    f32* restrict work,
    const INT lwork,
    INT* info)
{
    INT upper, lquery;
    INT iinfo, iws, j, k, kb, ldwork, lwkopt, nb, nbmin;

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
        *info = -7;
    }

    if (*info == 0) {
        nb = lapack_get_nb("SYTRF");
        lwkopt = (1 > n * nb) ? 1 : n * nb;
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SSYTRF_ROOK", -(*info));
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

                slasyf_rook(uplo, k + 1, nb, &kb, A, lda, ipiv, work, ldwork, &iinfo);

            } else {

                ssytf2_rook(uplo, k + 1, A, lda, ipiv, &iinfo);
                kb = k + 1;
            }

            if (*info == 0 && iinfo > 0) {
                *info = iinfo;
            }

            k = k - kb;
        }

    } else {

        k = 0;
        while (k < n) {

            if (k <= n - 1 - nb) {

                slasyf_rook(uplo, n - k, nb, &kb, &A[k + k * lda], lda,
                            &ipiv[k], work, ldwork, &iinfo);

            } else {

                ssytf2_rook(uplo, n - k, &A[k + k * lda], lda, &ipiv[k], &iinfo);
                kb = n - k;
            }

            if (*info == 0 && iinfo > 0) {
                *info = iinfo + k;
            }

            for (j = k; j < k + kb; j++) {
                if (ipiv[j] >= 0) {
                    ipiv[j] = ipiv[j] + k;
                } else {
                    ipiv[j] = ipiv[j] - k;
                }
            }

            k = k + kb;
        }
    }

    work[0] = (f32)lwkopt;
}
