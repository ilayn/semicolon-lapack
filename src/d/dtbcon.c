/**
 * @file dtbcon.c
 * @brief DTBCON estimates the reciprocal of the condition number of a triangular band matrix.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DTBCON estimates the reciprocal of the condition number of a
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
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void dtbcon(
    const char* norm,
    const char* uplo,
    const char* diag,
    const int n,
    const int kd,
    const double* const restrict AB,
    const int ldab,
    double* rcond,
    double* const restrict work,
    int* const restrict iwork,
    int* info)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int nounit, onenrm, upper;
    int ix, kase, kase1;
    double ainvnm, anorm, scale, smlnum, xnorm;
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
        xerbla("DTBCON", -(*info));
        return;
    }

    if (n == 0) {
        *rcond = ONE;
        return;
    }

    *rcond = ZERO;
    smlnum = dlamch("S") * ((1 > n) ? 1 : n);

    // Compute the norm of the triangular matrix A
    anorm = dlantb(norm, uplo, diag, n, kd, AB, ldab, work);

    // Continue only if anorm > 0
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
                    dlatbs(uplo, "N", diag, &normin, n, kd, AB, ldab, work, &scale, &work[2 * n], info);
                } else {
                    // Multiply by inv(A**T)
                    dlatbs(uplo, "T", diag, &normin, n, kd, AB, ldab, work, &scale, &work[2 * n], info);
                }
                normin = 'Y';

                // Multiply by 1/scale if doing so will not cause overflow
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
