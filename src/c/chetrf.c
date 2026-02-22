/**
 * @file chetrf.c
 * @brief CHETRF computes the factorization of a complex Hermitian matrix
 *        using the Bunch-Kaufman diagonal pivoting method.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * CHETRF computes the factorization of a complex Hermitian matrix A using
 * the Bunch-Kaufman diagonal pivoting method. The form of the
 * factorization is
 *
 *    A = U*D*U**H  or  A = L*D*L**H
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and D is Hermitian and block diagonal with
 * 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in]     uplo  = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A. If uplo = 'U', the
 *                      leading n-by-n upper triangular part of A contains the
 *                      upper triangular part of the matrix A, and the strictly
 *                      lower triangular part of A is not referenced. If
 *                      uplo = 'L', the leading n-by-n lower triangular part of
 *                      A contains the lower triangular part of the matrix A,
 *                      and the strictly upper triangular part of A is not
 *                      referenced.
 *                      On exit, the block diagonal matrix D and the multipliers
 *                      used to obtain the factor U or L (see below for further
 *                      details).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    ipiv  Integer array, dimension (n).
 *                      Details of the interchanges and the block structure of D.
 * @param[out]    work  Single complex array, dimension (max(1, lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The length of work. lwork >= 1. For best performance
 *                      lwork >= n*nb, where nb is the block size.
 *                      If lwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) is exactly zero.
 */
void chetrf(
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    c64* restrict work,
    const INT lwork,
    INT* info)
{
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    INT lquery = (lwork == -1);

    /* Test the input parameters */
    *info = 0;
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (lwork < 1 && !lquery) {
        *info = -7;
    }

    /* Determine the block size */
    INT nb = lapack_get_nb("HETRF");
    INT lwkopt = n * nb > 1 ? n * nb : 1;

    if (*info == 0) {
        work[0] = (c64)lwkopt;
    }

    if (*info != 0) {
        xerbla("CHETRF", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    INT nbmin = 2;
    INT ldwork = n;
    if (nb > 1 && nb < n) {
        INT iws = ldwork * nb;
        if (lwork < iws) {
            nb = lwork / ldwork;
            if (nb < 1) nb = 1;
            nbmin = lapack_get_nbmin("HETRF");
            if (nbmin < 2) nbmin = 2;
        }
    }
    if (nb < nbmin) {
        nb = n;
    }

    if (upper) {
        /* ============================================================
         * Factorize A as U*D*U**H using the upper triangle of A
         *
         * K is the main loop index, decreasing from n-1 to 0 in steps of
         * kb, where kb is the number of columns factorized by clahef.
         * ============================================================ */
        INT k = n - 1;
        while (k >= 0) {
            if (k + 1 > nb) {
                INT kb, iinfo;
                clahef(uplo, k + 1, nb, &kb, A, lda, ipiv, work, ldwork, &iinfo);

                /* Set info on the first occurrence of a zero pivot */
                if (*info == 0 && iinfo > 0) {
                    *info = iinfo;
                }

                k -= kb;
            } else {
                INT iinfo;
                chetf2(uplo, k + 1, A, lda, ipiv, &iinfo);

                /* Set info on the first occurrence of a zero pivot */
                if (*info == 0 && iinfo > 0) {
                    *info = iinfo;
                }

                k = -1;  /* exit loop */
            }
        }
    } else {
        /* ============================================================
         * Factorize A as L*D*L**H using the lower triangle of A
         *
         * K is the main loop index, increasing from 0 in steps of kb.
         * ============================================================ */
        INT k = 0;
        while (k < n) {
            if (k <= n - 1 - nb) {
                INT kb, iinfo;
                clahef(uplo, n - k, nb, &kb, &A[k + k * lda], lda,
                       &ipiv[k], work, ldwork, &iinfo);

                if (*info == 0 && iinfo > 0) {
                    *info = iinfo + k;
                }

                /* Adjust IPIV */
                for (INT j = k; j < k + kb; j++) {
                    if (ipiv[j] >= 0) {
                        ipiv[j] += k;
                    } else {
                        ipiv[j] -= k;
                    }
                }

                k += kb;
            } else {
                INT iinfo;
                chetf2(uplo, n - k, &A[k + k * lda], lda, &ipiv[k], &iinfo);

                if (*info == 0 && iinfo > 0) {
                    *info = iinfo + k;
                }

                /* Adjust IPIV for the unblocked portion */
                INT kb = n - k;
                for (INT j = k; j < k + kb; j++) {
                    if (ipiv[j] >= 0) {
                        ipiv[j] += k;
                    } else {
                        ipiv[j] -= k;
                    }
                }

                k = n;  /* exit loop */
            }
        }
    }

    work[0] = (c64)lwkopt;
}
