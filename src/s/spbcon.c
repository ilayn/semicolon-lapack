/**
 * @file spbcon.c
 * @brief SPBCON estimates the reciprocal condition number of a symmetric positive definite band matrix.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SPBCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite band matrix using the
 * Cholesky factorization A = U**T*U or A = L*L**T computed by SPBTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular factor stored in AB
 *                       - `'L'`: Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in]     AB     Array of dimension (`ldab`, `n`).
 *                       The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T of the band
 *                       matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of U or L is stored in the
 *                       j-th column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = U(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = L(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the symmetric band
 *                       matrix A.
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
void spbcon(
    const char* uplo,
    const INT n,
    const INT kd,
    const f32* restrict AB,
    const INT ldab,
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
    INT info_local;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (ldab < kd + 1) {
        *info = -5;
    } else if (anorm < ZERO) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SPBCON", -(*info));
        return;
    }

    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    smlnum = slamch("S");

    kase = 0;
    normin = 'N';
    do {
        slacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase != 0) {
            if (upper) {
                // Multiply by inv(U**T)
                slatbs("U", "T", "N", &normin, n, kd, AB, ldab, work, &scalel, &work[2 * n], &info_local);
                normin = 'Y';
                // Multiply by inv(U)
                slatbs("U", "N", "N", &normin, n, kd, AB, ldab, work, &scaleu, &work[2 * n], &info_local);
            } else {
                // Multiply by inv(L)
                slatbs("L", "N", "N", &normin, n, kd, AB, ldab, work, &scalel, &work[2 * n], &info_local);
                normin = 'Y';
                // Multiply by inv(L**T)
                slatbs("L", "T", "N", &normin, n, kd, AB, ldab, work, &scaleu, &work[2 * n], &info_local);
            }

            // Multiply by 1/scale if doing so will not cause overflow
            scale = scalel * scaleu;
            if (scale != ONE) {
                ix = cblas_isamax(n, work, 1);
                if (scale < fabsf(work[ix]) * smlnum || scale == ZERO)
                    return;
                srscl(n, scale, work, 1);
            }
        }
    } while (kase != 0);

    // Compute the estimate of the reciprocal condition number
    if (ainvnm != ZERO)
        *rcond = (ONE / ainvnm) / anorm;
}
