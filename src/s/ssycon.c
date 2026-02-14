/**
 * @file ssycon.c
 * @brief SSYCON estimates the reciprocal of the condition number of a
 *        real symmetric matrix using its factorization.
 */

#include "semicolon_lapack_single.h"

/**
 * SSYCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric matrix A using the factorization
 * A = U*D*U**T or A = L*D*L**T computed by SSYTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   Specifies whether the details of the factorization
 *                       are stored as an upper or lower triangular matrix.
 *                       = 'U': Upper triangular, form is A = U*D*U**T
 *                       = 'L': Lower triangular, form is A = L*D*L**T
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The block diagonal matrix D and the multipliers used
 *                       to obtain the factor U or L as computed by SSYTRF.
 *                       Double precision array, dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     ipiv   Details of the interchanges and the block structure
 *                       of D as determined by SSYTRF. Integer array, dimension (n).
 * @param[in]     anorm  The 1-norm of the original matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is
 *                       an estimate of the 1-norm of inv(A) computed in this routine.
 * @param[out]    work   Double precision array, dimension (2*n).
 * @param[out]    iwork  Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ssycon(
    const char* uplo,
    const int n,
    const f32* const restrict A,
    const int lda,
    const int* const restrict ipiv,
    const f32 anorm,
    f32* rcond,
    f32* const restrict work,
    int* const restrict iwork,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    // Test the input parameters.
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (anorm < ZERO) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SSYCON", -(*info));
        return;
    }

    // Quick return if possible.
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm <= ZERO) {
        return;
    }

    // Check that the diagonal matrix D is nonsingular.
    if (upper) {
        // Upper triangular storage: examine D from bottom to top.
        for (int i = n - 1; i >= 0; i--) {
            if (ipiv[i] >= 0 && A[i + i * lda] == ZERO) {
                return;
            }
        }
    } else {
        // Lower triangular storage: examine D from top to bottom.
        for (int i = 0; i < n; i++) {
            if (ipiv[i] >= 0 && A[i + i * lda] == ZERO) {
                return;
            }
        }
    }

    // Estimate the 1-norm of the inverse.
    int kase = 0;
    int isave[3] = {0, 0, 0};
    f32 ainvnm;
    int linfo;

    for (;;) {
        slacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase == 0) break;

        // Multiply by inv(L*D*L**T) or inv(U*D*U**T).
        ssytrs(uplo, n, 1, A, lda, ipiv, work, n, &linfo);
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
