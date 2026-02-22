/**
 * @file dtpcon.c
 * @brief DTPCON estimates the reciprocal condition number of a packed triangular matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DTPCON estimates the reciprocal of the condition number of a packed
 * triangular matrix A, in either the 1-norm or the infinity-norm.
 *
 * The norm of A is computed and an estimate is obtained for
 * norm(inv(A)), then the reciprocal of the condition number is
 * computed as RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 * @param[in]     norm   = '1' or 'O': 1-norm
 *                        = 'I': Infinity-norm
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The packed triangular matrix A. Array of dimension (n*(n+1)/2).
 * @param[out]    rcond  The reciprocal condition number.
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dtpcon(
    const char* norm,
    const char* uplo,
    const char* diag,
    const INT n,
    const f64* restrict AP,
    f64* rcond,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT nounit, onenrm, upper;
    char normin;
    INT ix, kase, kase1;
    f64 ainvnm, anorm, scale, smlnum, xnorm;
    INT isave[3];
    INT info_local;

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
    }
    if (*info != 0) {
        xerbla("DTPCON", -(*info));
        return;
    }

    if (n == 0) {
        *rcond = ONE;
        return;
    }

    *rcond = ZERO;
    smlnum = dlamch("S") * (f64)(1 > n ? 1 : n);

    // Compute the norm of the triangular matrix A
    anorm = dlantp(norm, uplo, diag, n, AP, work);

    // Continue only if ANORM > 0
    if (anorm > ZERO) {
        // Estimate the norm of the inverse of A
        ainvnm = ZERO;
        normin = 'N';
        if (onenrm) {
            kase1 = 1;
        } else {
            kase1 = 2;
        }
        kase = 0;
        do {
            dlacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
            if (kase != 0) {
                if (kase == kase1) {
                    // Multiply by inv(A)
                    dlatps(uplo, "N", diag, &normin, n, AP, work, &scale, &work[2 * n], &info_local);
                } else {
                    // Multiply by inv(A**T)
                    dlatps(uplo, "T", diag, &normin, n, AP, work, &scale, &work[2 * n], &info_local);
                }
                normin = 'Y';

                // Multiply by 1/SCALE if doing so will not cause overflow
                if (scale != ONE) {
                    ix = cblas_idamax(n, work, 1);
                    xnorm = fabs(work[ix]);
                    if (scale < xnorm * smlnum || scale == ZERO)
                        return;
                    drscl(n, scale, work, 1);
                }
            }
        } while (kase != 0);

        // Compute the estimate of the reciprocal condition number
        if (ainvnm != ZERO)
            *rcond = (ONE / anorm) / ainvnm;
    }
}
