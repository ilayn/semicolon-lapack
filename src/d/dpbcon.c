/**
 * @file dpbcon.c
 * @brief DPBCON estimates the reciprocal condition number of a symmetric positive definite band matrix.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DPBCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite band matrix using the
 * Cholesky factorization A = U**T*U or A = L*L**T computed by DPBTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangular factor stored in AB
 *                        = 'L': Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     AB     The triangular factor from DPBTRF. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in]     anorm  The 1-norm of the original matrix A.
 * @param[out]    rcond  The reciprocal condition number.
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dpbcon(
    const char* uplo,
    const INT n,
    const INT kd,
    const f64* restrict AB,
    const INT ldab,
    const f64 anorm,
    f64* rcond,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT upper;
    char normin;
    INT ix, kase;
    f64 ainvnm, scale, scalel, scaleu, smlnum;
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
        xerbla("DPBCON", -(*info));
        return;
    }

    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    smlnum = dlamch("S");

    kase = 0;
    normin = 'N';
    do {
        dlacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase != 0) {
            if (upper) {
                // Multiply by inv(U**T)
                dlatbs("U", "T", "N", &normin, n, kd, AB, ldab, work, &scalel, &work[2 * n], &info_local);
                normin = 'Y';
                // Multiply by inv(U)
                dlatbs("U", "N", "N", &normin, n, kd, AB, ldab, work, &scaleu, &work[2 * n], &info_local);
            } else {
                // Multiply by inv(L)
                dlatbs("L", "N", "N", &normin, n, kd, AB, ldab, work, &scalel, &work[2 * n], &info_local);
                normin = 'Y';
                // Multiply by inv(L**T)
                dlatbs("L", "T", "N", &normin, n, kd, AB, ldab, work, &scaleu, &work[2 * n], &info_local);
            }

            // Multiply by 1/scale if doing so will not cause overflow
            scale = scalel * scaleu;
            if (scale != ONE) {
                ix = cblas_idamax(n, work, 1);
                if (scale < fabs(work[ix]) * smlnum || scale == ZERO)
                    return;
                drscl(n, scale, work, 1);
            }
        }
    } while (kase != 0);

    // Compute the estimate of the reciprocal condition number
    if (ainvnm != ZERO)
        *rcond = (ONE / ainvnm) / anorm;
}
