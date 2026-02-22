/**
 * @file checon.c
 * @brief CHECON estimates the reciprocal of the condition number of a
 *        complex Hermitian matrix using its factorization.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CHECON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian matrix A using the factorization
 * A = U*D*U**H or A = L*D*L**H computed by CHETRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   Specifies whether the details of the factorization
 *                       are stored as an upper or lower triangular matrix.
 *                       = 'U': Upper triangular, form is A = U*D*U**H
 *                       = 'L': Lower triangular, form is A = L*D*L**H
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The block diagonal matrix D and the multipliers used
 *                       to obtain the factor U or L as computed by CHETRF.
 *                       Complex*16 array, dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     ipiv   Details of the interchanges and the block structure
 *                       of D as determined by CHETRF. Integer array, dimension (n).
 * @param[in]     anorm  The 1-norm of the original matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is
 *                       an estimate of the 1-norm of inv(A) computed in this routine.
 * @param[out]    work   Complex*16 array, dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void checon(
    const char* uplo,
    const INT n,
    const c64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    const f32 anorm,
    f32* rcond,
    c64* restrict work,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    // Test the input parameters.
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
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
        xerbla("CHECON", -(*info));
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
        for (INT i = n - 1; i >= 0; i--) {
            if (ipiv[i] >= 0 && A[i + i * lda] == ZERO) {
                return;
            }
        }
    } else {
        // Lower triangular storage: examine D from top to bottom.
        for (INT i = 0; i < n; i++) {
            if (ipiv[i] >= 0 && A[i + i * lda] == ZERO) {
                return;
            }
        }
    }

    // Estimate the 1-norm of the inverse.
    INT kase = 0;
    INT isave[3] = {0, 0, 0};
    f32 ainvnm;
    INT linfo;

    for (;;) {
        clacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase == 0) break;

        // Multiply by inv(L*D*L**H) or inv(U*D*U**H).
        chetrs(uplo, n, 1, A, lda, ipiv, work, n, &linfo);
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
