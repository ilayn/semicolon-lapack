/**
 * @file ctbcon.c
 * @brief CTBCON estimates the reciprocal of the condition number of a triangular band matrix.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CTBCON estimates the reciprocal of the condition number of a
 * triangular band matrix A, in either the 1-norm or the infinity-norm.
 *
 * The norm of A is computed and an estimate is obtained for
 * norm(inv(A)), then the reciprocal of the condition number is
 * computed as
 *    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 * @param[in]     norm   = '1' or 'O': 1-norm
 *                        = 'I': Infinity-norm
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of superdiagonals or subdiagonals. kd >= 0.
 * @param[in]     AB     The triangular band matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    rcond  The reciprocal of the condition number.
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Single precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ctbcon(
    const char* norm,
    const char* uplo,
    const char* diag,
    const int n,
    const int kd,
    const c64* restrict AB,
    const int ldab,
    f32* rcond,
    c64* restrict work,
    f32* restrict rwork,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    int nounit, onenrm, upper;
    int ix, kase, kase1;
    f32 ainvnm, anorm, scale, smlnum, xnorm;
    int isave[3];
    char normin;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    onenrm = (norm[0] == '1' || norm[0] == 'O' || norm[0] == 'o');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

    if (!onenrm && !(norm[0] == 'I' || norm[0] == 'i')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (kd < 0) {
        *info = -5;
    } else if (ldab < kd + 1) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("CTBCON", -(*info));
        return;
    }

    if (n == 0) {
        *rcond = ONE;
        return;
    }

    *rcond = ZERO;
    smlnum = slamch("S") * ((1 > n) ? 1 : n);

    anorm = clantb(norm, uplo, diag, n, kd, AB, ldab, rwork);

    if (anorm > ZERO) {
        ainvnm = ZERO;
        normin = 'N';
        if (onenrm) {
            kase1 = 1;
        } else {
            kase1 = 2;
        }
        kase = 0;
        do {
            clacn2(n, &work[n], work, &ainvnm, &kase, isave);
            if (kase != 0) {
                if (kase == kase1) {
                    clatbs(uplo, "N", diag, &normin, n, kd, AB, ldab, work, &scale, rwork, info);
                } else {
                    clatbs(uplo, "C", diag, &normin, n, kd, AB, ldab, work, &scale, rwork, info);
                }
                normin = 'Y';

                if (scale != ONE) {
                    ix = cblas_icamax(n, work, 1);
                    xnorm = cabs1f(work[ix]);
                    if (scale < xnorm * smlnum || scale == ZERO)
                        return;
                    cdrscl(n, scale, work, 1);
                }
            }
        } while (kase != 0);

        if (ainvnm != ZERO)
            *rcond = (ONE / anorm) / ainvnm;
    }
}
