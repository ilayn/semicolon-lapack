/**
 * @file zpbcon.c
 * @brief ZPBCON estimates the reciprocal condition number of a Hermitian positive definite band matrix.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite band matrix using
 * the Cholesky factorization A = U**H*U or A = L*L**H computed by
 * ZPBTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangular factor stored in AB
 *                        = 'L': Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     AB     The triangular factor from ZPBTRF. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the Hermitian band matrix A.
 * @param[out]    rcond  The reciprocal condition number.
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zpbcon(
    const char* uplo,
    const int n,
    const int kd,
    const c128* const restrict AB,
    const int ldab,
    const f64 anorm,
    f64* rcond,
    c128* const restrict work,
    f64* const restrict rwork,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int upper;
    char normin;
    int ix, kase;
    f64 ainvnm, scale, scalel, scaleu, smlnum;
    int isave[3];

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
        xerbla("ZPBCON", -(*info));
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
        zlacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase != 0) {
            if (upper) {
                /* Multiply by inv(U**H) */
                zlatbs("U", "C", "N", &normin, n, kd, AB, ldab, work, &scalel, rwork, info);
                normin = 'Y';
                /* Multiply by inv(U) */
                zlatbs("U", "N", "N", &normin, n, kd, AB, ldab, work, &scaleu, rwork, info);
            } else {
                /* Multiply by inv(L) */
                zlatbs("L", "N", "N", &normin, n, kd, AB, ldab, work, &scalel, rwork, info);
                normin = 'Y';
                /* Multiply by inv(L**H) */
                zlatbs("L", "C", "N", &normin, n, kd, AB, ldab, work, &scaleu, rwork, info);
            }

            /* Multiply by 1/SCALE if doing so will not cause overflow */
            scale = scalel * scaleu;
            if (scale != ONE) {
                ix = cblas_izamax(n, work, 1);
                if (scale < cabs1(work[ix]) * smlnum || scale == ZERO)
                    return;
                zdrscl(n, scale, work, 1);
            }
        }
    } while (kase != 0);

    /* Compute the estimate of the reciprocal condition number */
    if (ainvnm != ZERO)
        *rcond = (ONE / ainvnm) / anorm;
}
