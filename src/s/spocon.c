/**
 * @file spocon.c
 * @brief SPOCON estimates the reciprocal of the condition number of a
 *        symmetric positive definite matrix.
 */

#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SPOCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite matrix using the
 * Cholesky factorization A = U**T*U or A = L*L**T computed by SPOTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, as computed
 *                       by spotrf. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the symmetric matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM).
 * @param[out]    work   Double precision array, dimension (3*n).
 * @param[out]    iwork  Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void spocon(
    const char* uplo,
    const INT n,
    const f32* restrict A,
    const INT lda,
    const f32 anorm,
    f32* rcond,
    f32* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    // Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (anorm < ZERO) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("SPOCON", -(*info));
        return;
    }

    // Quick return if possible
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    // Safe minimum
    f32 smlnum = slamch("S");

    // Estimate the 1-norm of inv(A).
    INT kase = 0;
    char normin = 'N';
    INT isave[3] = {0, 0, 0};
    f32 ainvnm;

    for (;;) {
        slacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase == 0) break;

        f32 scalel, scaleu;
        INT linfo;

        if (upper) {
            // Multiply by inv(U**T).
            slatrs("U", "T", "N", &normin, n, A, lda, work, &scalel,
                   &work[2 * n], &linfo);
            normin = 'Y';

            // Multiply by inv(U).
            slatrs("U", "N", "N", &normin, n, A, lda, work, &scaleu,
                   &work[2 * n], &linfo);
        } else {
            // Multiply by inv(L).
            slatrs("L", "N", "N", &normin, n, A, lda, work, &scalel,
                   &work[2 * n], &linfo);
            normin = 'Y';

            // Multiply by inv(L**T).
            slatrs("L", "T", "N", &normin, n, A, lda, work, &scaleu,
                   &work[2 * n], &linfo);
        }

        // Multiply by 1/SCALE if doing so will not cause overflow.
        f32 scale = scalel * scaleu;
        if (scale != ONE) {
            INT ix = cblas_isamax(n, work, 1);
            if (scale < fabsf(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            srscl(n, scale, work, 1);
        }
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
