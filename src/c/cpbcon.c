/**
 * @file cpbcon.c
 * @brief CPBCON estimates the reciprocal condition number of a Hermitian positive definite band matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPBCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite band matrix using
 * the Cholesky factorization A = U**H*U or A = L*L**H computed by
 * CPBTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangular factor stored in AB
 *                        = 'L': Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     AB     The triangular factor from CPBTRF. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the Hermitian band matrix A.
 * @param[out]    rcond  The reciprocal condition number.
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cpbcon(
    const char* uplo,
    const INT n,
    const INT kd,
    const c64* restrict AB,
    const INT ldab,
    const f32 anorm,
    f32* rcond,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT upper;
    char normin;
    INT ix, kase;
    f32 ainvnm, scale, scalel, scaleu, smlnum;
    INT isave[3];

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
        xerbla("CPBCON", -(*info));
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
        clacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase != 0) {
            if (upper) {
                /* Multiply by inv(U**H) */
                clatbs("U", "C", "N", &normin, n, kd, AB, ldab, work, &scalel, rwork, info);
                normin = 'Y';
                /* Multiply by inv(U) */
                clatbs("U", "N", "N", &normin, n, kd, AB, ldab, work, &scaleu, rwork, info);
            } else {
                /* Multiply by inv(L) */
                clatbs("L", "N", "N", &normin, n, kd, AB, ldab, work, &scalel, rwork, info);
                normin = 'Y';
                /* Multiply by inv(L**H) */
                clatbs("L", "C", "N", &normin, n, kd, AB, ldab, work, &scaleu, rwork, info);
            }

            /* Multiply by 1/SCALE if doing so will not cause overflow */
            scale = scalel * scaleu;
            if (scale != ONE) {
                ix = cblas_icamax(n, work, 1);
                if (scale < cabs1f(work[ix]) * smlnum || scale == ZERO)
                    return;
                cdrscl(n, scale, work, 1);
            }
        }
    } while (kase != 0);

    /* Compute the estimate of the reciprocal condition number */
    if (ainvnm != ZERO)
        *rcond = (ONE / ainvnm) / anorm;
}
