/**
 * @file dppcon.c
 * @brief DPPCON estimates the reciprocal of the condition number of a symmetric positive definite matrix in packed storage.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPPCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite packed matrix using
 * the Cholesky factorization A = U**T*U or A = L*L**T computed by
 * DPPTRF.
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
void dppcon(
    const char* uplo,
    const int n,
    const f64* const restrict AP,
    const f64 anorm,
    f64* rcond,
    f64* const restrict work,
    int* const restrict iwork,
    int* info)
{
    // dppcon.f lines 136-137: Parameters
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // dppcon.f lines 140-146: Local Scalars and Arrays
    int upper;
    char normin;
    int ix, kase;
    f64 ainvnm, scale, scalel, scaleu, smlnum;
    int isave[3];

    // dppcon.f lines 164-176: Test the input parameters
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
        xerbla("DPPCON", -(*info));
        return;
    }

    // dppcon.f lines 180-186: Quick return if possible
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    // dppcon.f line 188
    smlnum = dlamch("S");  // Safe minimum

    // dppcon.f lines 192-193: Initialize for norm estimation
    kase = 0;
    normin = 'N';

    // dppcon.f lines 194-235: Main iteration loop
    while (1) {
        // dppcon.f line 195: CALL DLACN2( N, WORK( N+1 ), WORK, IWORK, AINVNM, KASE, ISAVE )
        dlacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);

        // dppcon.f line 196
        if (kase == 0) {
            break;
        }

        if (upper) {
            // dppcon.f lines 199-202: Multiply by inv(U**T)
            dlatps("U", "T", "N", &normin, n, AP, work, &scalel, &work[2 * n], info);
            // dppcon.f line 203
            normin = 'Y';

            // dppcon.f lines 207-209: Multiply by inv(U)
            dlatps("U", "N", "N", &normin, n, AP, work, &scaleu, &work[2 * n], info);
        } else {
            // dppcon.f lines 214-216: Multiply by inv(L)
            dlatps("L", "N", "N", &normin, n, AP, work, &scalel, &work[2 * n], info);
            // dppcon.f line 217
            normin = 'Y';

            // dppcon.f lines 221-222: Multiply by inv(L**T)
            dlatps("L", "T", "N", &normin, n, AP, work, &scaleu, &work[2 * n], info);
        }

        // dppcon.f lines 227-233: Multiply by 1/SCALE if doing so will not cause overflow
        scale = scalel * scaleu;
        if (scale != ONE) {
            // dppcon.f line 229: IX = IDAMAX( N, WORK, 1 )
            // CBLAS idamax returns 0-based index; Fortran IDAMAX returns 1-based
            ix = cblas_idamax(n, work, 1);
            // dppcon.f lines 230-231
            if (scale < fabs(work[ix]) * smlnum || scale == ZERO) {
                return;  // GO TO 20 (exit)
            }
            // dppcon.f line 232
            drscl(n, scale, work, 1);
        }
        // dppcon.f line 234: GO TO 10 (continue loop)
    }

    // dppcon.f lines 239-240: Compute the estimate of the reciprocal condition number
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
