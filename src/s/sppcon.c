/**
 * @file sppcon.c
 * @brief SPPCON estimates the reciprocal of the condition number of a symmetric positive definite matrix in packed storage.
 */

#include <math.h>
#include <cblas.h>
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
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, packed
 *                       columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the symmetric matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sppcon(
    const char* uplo,
    const int n,
    const f32* const restrict AP,
    const f32 anorm,
    f32* rcond,
    f32* const restrict work,
    int* const restrict iwork,
    int* info)
{
    // sppcon.f lines 136-137: Parameters
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    // sppcon.f lines 140-146: Local Scalars and Arrays
    int upper;
    char normin;
    int ix, kase;
    f32 ainvnm, scale, scalel, scaleu, smlnum;
    int isave[3];

    // sppcon.f lines 164-176: Test the input parameters
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

    // sppcon.f lines 180-186: Quick return if possible
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    // sppcon.f line 188
    smlnum = slamch("S");  // Safe minimum

    // sppcon.f lines 192-193: Initialize for norm estimation
    kase = 0;
    normin = 'N';

    // sppcon.f lines 194-235: Main iteration loop
    while (1) {
        // sppcon.f line 195: CALL SLACN2( N, WORK( N+1 ), WORK, IWORK, AINVNM, KASE, ISAVE )
        slacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);

        // sppcon.f line 196
        if (kase == 0) {
            break;
        }

        if (upper) {
            // sppcon.f lines 199-202: Multiply by inv(U**T)
            slatps("U", "T", "N", &normin, n, AP, work, &scalel, &work[2 * n], info);
            // sppcon.f line 203
            normin = 'Y';

            // sppcon.f lines 207-209: Multiply by inv(U)
            slatps("U", "N", "N", &normin, n, AP, work, &scaleu, &work[2 * n], info);
        } else {
            // sppcon.f lines 214-216: Multiply by inv(L)
            slatps("L", "N", "N", &normin, n, AP, work, &scalel, &work[2 * n], info);
            // sppcon.f line 217
            normin = 'Y';

            // sppcon.f lines 221-222: Multiply by inv(L**T)
            slatps("L", "T", "N", &normin, n, AP, work, &scaleu, &work[2 * n], info);
        }

        // sppcon.f lines 227-233: Multiply by 1/SCALE if doing so will not cause overflow
        scale = scalel * scaleu;
        if (scale != ONE) {
            // sppcon.f line 229: IX = IDAMAX( N, WORK, 1 )
            // CBLAS idamax returns 0-based index; Fortran IDAMAX returns 1-based
            ix = cblas_isamax(n, work, 1);
            // sppcon.f lines 230-231
            if (scale < fabsf(work[ix]) * smlnum || scale == ZERO) {
                return;  // GO TO 20 (exit)
            }
            // sppcon.f line 232
            srscl(n, scale, work, 1);
        }
        // sppcon.f line 234: GO TO 10 (continue loop)
    }

    // sppcon.f lines 239-240: Compute the estimate of the reciprocal condition number
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
