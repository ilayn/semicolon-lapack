/**
 * @file csytrf.c
 * @brief CSYTRF computes the factorization of a complex symmetric matrix
 *        using the Bunch-Kaufman diagonal pivoting method.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>

/**
 * CSYTRF computes the factorization of a complex symmetric matrix A using
 * the Bunch-Kaufman diagonal pivoting method. The form of the
 * factorization is
 *
 *    A = U*D*U**T  or  A = L*D*L**T
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and D is symmetric and block diagonal with
 * 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in]     uplo  = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the symmetric matrix A.
 *                      On exit, the block diagonal matrix D and the
 *                      multipliers used to obtain the factor U or L.
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
void csytrf(
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
    INT nb = lapack_get_nb("SYTRF");
    INT lwkopt = n * nb > 1 ? n * nb : 1;

    if (*info == 0) {
        work[0] = (c64)lwkopt;
    }

    if (*info != 0) {
        xerbla("CSYTRF", -(*info));
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
            nbmin = lapack_get_nbmin("SYTRF");
            if (nbmin < 2) nbmin = 2;
        }
    }
    if (nb < nbmin) {
        nb = n;
    }

    if (upper) {
        /* ============================================================
         * Factorize A as U*D*U**T using the upper triangle of A
         *
         * K is the main loop index, decreasing from n-1 to 0 in steps of
         * kb, where kb is the number of columns factorized by clasyf.
         * ============================================================ */
        INT k = n - 1;
        while (k >= 0) {
            if (k + 1 > nb) {
                /* Factorize columns of the leading (k+1)-by-(k+1) submatrix
                 * using blocked code.
                 *
                 * Fortran: CLASYF(UPLO, K, NB, KB, A, LDA, IPIV, WORK, LDWORK, IINFO)
                 * Fortran K (1-based) = k+1 (the submatrix size).
                 * clasyf operates on A(0:k, 0:k) and factors the trailing nb columns. */
                INT kb, iinfo;
                clasyf(uplo, k + 1, nb, &kb, A, lda, ipiv, work, ldwork, &iinfo);

                /* Set info on the first occurrence of a zero pivot */
                if (*info == 0 && iinfo > 0) {
                    *info = iinfo;
                }

                k -= kb;
            } else {
                /* Use unblocked code to factorize the remaining (k+1) columns.
                 *
                 * Fortran: CSYTF2(UPLO, K, A, LDA, IPIV, IINFO)
                 * Operates on A(0:k, 0:k). */
                INT iinfo;
                csytf2(uplo, k + 1, A, lda, ipiv, &iinfo);

                /* Set info on the first occurrence of a zero pivot */
                if (*info == 0 && iinfo > 0) {
                    *info = iinfo;
                }

                k = -1;  /* exit loop */
            }
        }
    } else {
        /* ============================================================
         * Factorize A as L*D*L**T using the lower triangle of A
         *
         * K is the main loop index, increasing from 0 in steps of kb.
         * ============================================================ */
        INT k = 0;
        while (k < n) {
            if (k <= n - 1 - nb) {
                /* Factorize columns k:k+kb-1 of A using blocked code.
                 *
                 * Fortran: CLASYF(UPLO, N-K+1, NB, KB, A(K,K), LDA, IPIV(K), WORK, LDWORK, IINFO)
                 * Submatrix size = n-k. Operates on A(k:n-1, k:n-1). */
                INT kb, iinfo;
                clasyf(uplo, n - k, nb, &kb, &A[k + k * lda], lda,
                       &ipiv[k], work, ldwork, &iinfo);

                /* Set info on the first occurrence of a zero pivot.
                 * Fortran: INFO = IINFO + K - 1. In 0-based: info = iinfo + k
                 * (since iinfo is already 1-based from clasyf, and k is the
                 * 0-based offset, the global 1-based index = iinfo + k). */
                if (*info == 0 && iinfo > 0) {
                    *info = iinfo + k;
                }

                /* Adjust IPIV: clasyf returns local indices for the submatrix
                 * starting at column k. We need global indices.
                 *
                 * Fortran: IPIV(J) = IPIV(J) + K - 1 (positive)
                 *          IPIV(J) = IPIV(J) - K + 1 (negative)
                 * 0-based: positive ipiv[j] += k
                 *          negative: ipiv[j] is -(local+1), need -(global+1)
                 *          where global = local + k, so -(global+1) = -(local+k+1)
                 *          = ipiv[j] - k (since ipiv[j] = -(local+1)) */
                for (INT j = k; j < k + kb; j++) {
                    if (ipiv[j] >= 0) {
                        ipiv[j] += k;
                    } else {
                        ipiv[j] -= k;
                    }
                }

                k += kb;
            } else {
                /* Use unblocked code to factorize the remaining columns k:n-1.
                 *
                 * Fortran: CSYTF2(UPLO, N-K+1, A(K,K), LDA, IPIV(K), IINFO)
                 * Submatrix size = n-k. */
                INT iinfo;
                csytf2(uplo, n - k, &A[k + k * lda], lda, &ipiv[k], &iinfo);

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
