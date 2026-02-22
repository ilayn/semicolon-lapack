/**
 * @file cpocon.c
 * @brief CPOCON estimates the reciprocal of the condition number of a
 *        complex Hermitian positive definite matrix.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPOCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite matrix using the
 * Cholesky factorization A = U**H*U or A = L*L**H computed by CPOTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, as computed
 *                       by cpotrf. Complex array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the Hermitian matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM).
 * @param[out]    work   Complex array, dimension (2*n).
 * @param[out]    rwork  Single precision array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void cpocon(
    const char* uplo,
    const INT n,
    const c64* restrict A,
    const INT lda,
    const f32 anorm,
    f32* rcond,
    c64* restrict work,
    f32* restrict rwork,
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
        xerbla("CPOCON", -(*info));
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

    f32 smlnum = slamch("S");

    // Estimate the 1-norm of inv(A).
    INT kase = 0;
    char normin = 'N';
    INT isave[3] = {0, 0, 0};
    f32 ainvnm;

    for (;;) {
        clacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase == 0) break;

        f32 scalel, scaleu;
        INT linfo;

        if (upper) {
            // Multiply by inv(U**H).
            clatrs("U", "C", "N", &normin, n, A, lda, work, &scalel,
                   rwork, &linfo);
            normin = 'Y';

            // Multiply by inv(U).
            clatrs("U", "N", "N", &normin, n, A, lda, work, &scaleu,
                   rwork, &linfo);
        } else {
            // Multiply by inv(L).
            clatrs("L", "N", "N", &normin, n, A, lda, work, &scalel,
                   rwork, &linfo);
            normin = 'Y';

            // Multiply by inv(L**H).
            clatrs("L", "C", "N", &normin, n, A, lda, work, &scaleu,
                   rwork, &linfo);
        }

        // Multiply by 1/SCALE if doing so will not cause overflow.
        f32 scale = scalel * scaleu;
        if (scale != ONE) {
            INT ix = cblas_icamax(n, work, 1);
            if (scale < cabs1f(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            cdrscl(n, scale, work, 1);
        }
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
