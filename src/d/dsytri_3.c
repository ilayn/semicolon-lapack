/**
 * @file dsytri_3.c
 * @brief DSYTRI_3 computes the inverse of a real symmetric indefinite matrix using the factorization computed by DSYTRF_RK or DSYTRF_BK.
 */

#include "semicolon_lapack_double.h"

/**
 * DSYTRI_3 computes the inverse of a real symmetric indefinite
 * matrix A using the factorization computed by DSYTRF_RK or DSYTRF_BK:
 *
 *     A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * DSYTRI_3 sets the leading dimension of the workspace before calling
 * DSYTRI_3X that actually computes the inverse. This is the blocked
 * version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are
 *          stored as an upper or lower triangular matrix.
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, diagonal of the block diagonal matrix D and
 *          factors U or L as computed by DSYTRF_RK and DSYTRF_BK.
 *          On exit, if info = 0, the symmetric inverse of the original
 *          matrix.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] E
 *          Double precision array, dimension (n).
 *          Contains the superdiagonal (or subdiagonal) elements of the
 *          symmetric block diagonal matrix D.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
 *
 * @param[out] work
 *          Double precision array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The length of work.
 *          If n = 0, lwork >= 1, else lwork >= (n+nb+1)*(nb+3).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void dsytri_3(
    const char* uplo,
    const int n,
    f64* restrict A,
    const int lda,
    const f64* restrict E,
    const int* restrict ipiv,
    f64* restrict work,
    const int lwork,
    int* info)
{
    int upper, lquery;
    int lwkopt, nb;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    if (n == 0) {
        lwkopt = 1;
    } else {
        nb = 1;
        if (nb < 1) {
            nb = 1;
        }
        lwkopt = (n + nb + 1) * (nb + 3);
    }
    work[0] = (f64)lwkopt;

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (lwork < lwkopt && !lquery) {
        *info = -8;
    }

    if (*info != 0) {
        xerbla("DSYTRI_3", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    dsytri_3x(uplo, n, A, lda, E, ipiv, work, nb, info);

    work[0] = (f64)lwkopt;
}
