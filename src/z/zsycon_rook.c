/**
 * @file zsycon_rook.c
 * @brief ZSYCON_ROOK estimates the reciprocal of the condition number of a complex symmetric matrix using the factorization computed by ZSYTRF_ROOK.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZSYCON_ROOK estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex symmetric matrix A using the factorization
 * A = U*D*U**T or A = L*D*L**T computed by ZSYTRF_ROOK.
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
 *          Complex*16 array, dimension (lda, n).
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by ZSYTRF_ROOK.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by ZSYTRF_ROOK.
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
 *          Complex*16 array, dimension (2*n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zsycon_rook(
    const char* uplo,
    const int n,
    const c128* restrict A,
    const int lda,
    const int* restrict ipiv,
    const f64 anorm,
    f64* rcond,
    c128* restrict work,
    int* info)
{
    int upper;
    int i, kase;
    f64 ainvnm;
    int isave[3];
    int dummy_info;

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
        xerbla("ZSYCON_ROOK", -(*info));
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
            if (ipiv[i] >= 0 && A[i + i * lda] == CMPLX(0.0, 0.0)) {
                return;
            }
        }

    } else {

        for (i = 0; i < n; i++) {
            if (ipiv[i] >= 0 && A[i + i * lda] == CMPLX(0.0, 0.0)) {
                return;
            }
        }
    }

    kase = 0;
    for (;;) {
        zlacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase == 0) {
            break;
        }

        zsytrs_rook(uplo, n, 1, A, lda, ipiv, work, n, &dummy_info);
    }

    if (ainvnm != 0.0) {
        *rcond = (1.0 / ainvnm) / anorm;
    }
}
