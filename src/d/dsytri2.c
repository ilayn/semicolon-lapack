/**
 * @file dsytri2.c
 * @brief DSYTRI2 computes the inverse of a DOUBLE PRECISION symmetric indefinite
 *        matrix A using the factorization computed by DSYTRF.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"

/**
 * DSYTRI2 computes the inverse of a DOUBLE PRECISION symmetric indefinite matrix
 * A using the factorization A = U*D*U**T or A = L*D*L**T computed by
 * DSYTRF. DSYTRI2 sets the LEADING DIMENSION of the workspace
 * before calling DSYTRI2X that actually computes the inverse.
 *
 * @param[in]     uplo  Specifies whether the details of the factorization
 *                      are stored as an upper or lower triangular matrix.
 *                      = 'U': Upper triangular, form is A = U*D*U**T;
 *                      = 'L': Lower triangular, form is A = L*D*L**T.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the block diagonal matrix D and the multipliers
 *                      used to obtain the factor U or L as computed by DSYTRF.
 *                      On exit, if info = 0, the (symmetric) inverse of the
 *                      original matrix.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     ipiv  Integer array, dimension (n).
 *                      Details of the interchanges and the block structure of D
 *                      as determined by DSYTRF.
 * @param[out]    work  Double precision array, dimension (max(1, lwork)).
 * @param[in]     lwork The dimension of the array work.
 *                      If n = 0, lwork >= 1, else lwork >= (n+nb+1)*(nb+3).
 *                      If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void dsytri2(
    const char* uplo,
    const int n,
    f64* restrict A,
    const int lda,
    const int* restrict ipiv,
    f64* restrict work,
    const int lwork,
    int* info)
{
    int upper, lquery;
    int minsize, nbmax;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    nbmax = lapack_get_nb("SYTRI2");

    if (n == 0) {
        minsize = 1;
    } else if (nbmax >= n) {
        minsize = n;
    } else {
        minsize = (n + nbmax + 1) * (nbmax + 3);
    }

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (lwork < minsize && !lquery) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("DSYTRI2", -(*info));
        return;
    } else if (lquery) {
        work[0] = (f64)minsize;
        return;
    }

    if (n == 0) {
        return;
    }

    if (nbmax >= n) {
        dsytri(uplo, n, A, lda, ipiv, work, info);
    } else {
        dsytri2x(uplo, n, A, lda, ipiv, work, nbmax, info);
    }
}
