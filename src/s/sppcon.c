/**
 * @file sppcon.c
 * @brief SPPCON estimates the reciprocal of the condition number of a symmetric positive definite matrix in packed storage.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SPPCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite packed matrix using
 * the Cholesky factorization A = U**T*U or A = L*L**T computed by
 * SPPTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     AP     Array of dimension `n*(n+1)/2`.
 *                       The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, packed
 *                       columnwise in a linear array. The j-th column of
 *                       U or L is stored in the array AP as follows:
 *                       if `uplo='U'`, `AP[i + j*(j+1)/2] = U(i,j)` for `0<=i<=j`;
 *                       if `uplo='L'`, `AP[i + j*(2*n-j-1)/2] = L(i,j)` for
 *                       `j<=i<=n-1`.
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the symmetric matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as `rcond=1/(anorm*ainvnm)`, where `ainvnm` is
 *                       an estimate of the 1-norm of inv(A) computed in this
 *                       routine.
 * @param[out]    work   Array of dimension `3*n`.
 * @param[out]    iwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void sppcon(
    const char* uplo,
    const INT n,
    const f32* restrict AP,
    const f32 anorm,
    f32* rcond,
    f32* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT upper;
    char normin;
    INT ix, kase;
    f32 ainvnm, scale, scalel, scaleu, smlnum;
    INT isave[3];

    // Test the input parameters
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (anorm < ZERO) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SPPCON", -(*info));
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

    smlnum = slamch("S");

    // Estimate the 1-norm of the inverse.
    kase = 0;
    normin = 'N';

    while (1) {
        slacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (upper) {
            // Multiply by inv(U**T)
            slatps("U", "T", "N", &normin, n, AP, work, &scalel, &work[2 * n], info);
            normin = 'Y';

            // Multiply by inv(U)
            slatps("U", "N", "N", &normin, n, AP, work, &scaleu, &work[2 * n], info);
        } else {
            // Multiply by inv(L)
            slatps("L", "N", "N", &normin, n, AP, work, &scalel, &work[2 * n], info);
            normin = 'Y';

            // Multiply by inv(L**T)
            slatps("L", "T", "N", &normin, n, AP, work, &scaleu, &work[2 * n], info);
        }

        // Multiply by 1/SCALE if doing so will not cause overflow
        scale = scalel * scaleu;
        if (scale != ONE) {
            ix = cblas_isamax(n, work, 1);
            if (scale < fabsf(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            srscl(n, scale, work, 1);
        }
    }

    // Compute the estimate of the reciprocal condition number
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
