/**
 * @file dsycon_rook.c
 * @brief DSYCON_ROOK estimates the reciprocal of the condition number of a symmetric matrix using the factorization computed by DSYTRF_ROOK.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"

/**
 * DSYCON_ROOK estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric matrix A using the factorization
 * A = U*D*U**T or A = L*D*L**T computed by DSYTRF_ROOK.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] A
 *          Double precision array, dimension (lda, n).
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by DSYTRF_ROOK.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by DSYTRF_ROOK.
 *
 * @param[in] anorm
 *          The 1-norm of the original matrix A.
 *
 * @param[out] rcond
 *          The reciprocal of the condition number of the matrix A,
 *          computed as rcond = 1/(anorm * ainvnm), where ainvnm is an
 *          estimate of the 1-norm of inv(A) computed in this routine.
 *
 * @param[out] work
 *          Double precision array, dimension (2*n).
 *
 * @param[out] iwork
 *          Integer array, dimension (n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dsycon_rook(
    const char* uplo,
    const INT n,
    const f64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    const f64 anorm,
    f64* rcond,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    INT upper;
    INT i, kase;
    f64 ainvnm;
    INT isave[3];
    INT dummy_info;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (anorm < 0.0) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("DSYCON_ROOK", -(*info));
        return;
    }

    *rcond = 0.0;
    if (n == 0) {
        *rcond = 1.0;
        return;
    } else if (anorm <= 0.0) {
        return;
    }

    if (upper) {

        for (i = n - 1; i >= 0; i--) {
            if (ipiv[i] >= 0 && A[i + i * lda] == 0.0) {
                return;
            }
        }

    } else {

        for (i = 0; i < n; i++) {
            if (ipiv[i] >= 0 && A[i + i * lda] == 0.0) {
                return;
            }
        }
    }

    kase = 0;
    for (;;) {
        dlacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase == 0) {
            break;
        }

        dsytrs_rook(uplo, n, 1, A, lda, ipiv, work, n, &dummy_info);
    }

    if (ainvnm != 0.0) {
        *rcond = (1.0 / ainvnm) / anorm;
    }
}
