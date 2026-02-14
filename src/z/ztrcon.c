/** @file ztrcon.c
 * @brief ZTRCON estimates the reciprocal condition number of a triangular matrix. */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTRCON estimates the reciprocal of the condition number of a
 * triangular matrix A, in either the 1-norm or the infinity-norm.
 *
 * The norm of A is computed and an estimate is obtained for
 * norm(inv(A)), then the reciprocal of the condition number is
 * computed as
 *    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
 *
 * @param[in]     norm   Specifies whether the 1-norm condition number or the
 *                       infinity-norm condition number is required:
 *                       - '1' or 'O': 1-norm
 *                       - 'I': Infinity-norm
 * @param[in]     uplo   'U': A is upper triangular; 'L': A is lower triangular.
 * @param[in]     diag   'N': A is non-unit triangular; 'U': A is unit triangular.
 * @param[in]     n      The order of the matrix A (n >= 0).
 * @param[in]     A      The triangular matrix A. Array of dimension (lda, n).
 *                       If uplo = 'U', the leading n-by-n upper triangular part
 *                       contains the upper triangular matrix.
 *                       If uplo = 'L', the leading n-by-n lower triangular part
 *                       contains the lower triangular matrix.
 *                       If diag = 'U', the diagonal elements are not referenced
 *                       and are assumed to be 1.
 * @param[in]     lda    The leading dimension of the array A (lda >= max(1,n)).
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(norm(A) * norm(inv(A))).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ztrcon(const char* norm, const char* uplo, const char* diag,
            const int n, const c128* restrict A, const int lda,
            f64* rcond, c128* restrict work,
            f64* restrict rwork, int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int upper, onenrm, nounit;
    char normin;
    int ix, kase, kase1;
    f64 ainvnm, anorm, scale, smlnum, xnorm;
    int isave[3];

    /* Test the input parameters */
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    onenrm = (norm[0] == '1' || norm[0] == 'O' || norm[0] == 'o');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

    if (!onenrm && norm[0] != 'I' && norm[0] != 'i') {
        *info = -1;
    } else if (!upper && uplo[0] != 'L' && uplo[0] != 'l') {
        *info = -2;
    } else if (!nounit && diag[0] != 'U' && diag[0] != 'u') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    }

    if (*info != 0) {
        xerbla("ZTRCON", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        *rcond = ONE;
        return;
    }

    *rcond = ZERO;
    smlnum = dlamch("S") * (f64)(n > 1 ? n : 1);

    /* Compute the norm of the triangular matrix A */
    anorm = zlantr(norm, uplo, diag, n, n, A, lda, rwork);

    /* Continue only if anorm > 0 */
    if (anorm > ZERO) {

        /* Estimate the norm of the inverse of A */
        ainvnm = ZERO;
        normin = 'N';
        if (onenrm) {
            kase1 = 1;
        } else {
            kase1 = 2;
        }
        kase = 0;

        while (1) {
            zlacn2(n, &work[n], work, &ainvnm, &kase, isave);

            if (kase == 0) {
                break;
            }

            if (kase == kase1) {
                /* Multiply by inv(A) */
                zlatrs(uplo, "N", diag, &normin, n, A, lda,
                       work, &scale, rwork, info);
            } else {
                /* Multiply by inv(A**H) */
                zlatrs(uplo, "C", diag, &normin, n, A, lda,
                       work, &scale, rwork, info);
            }
            normin = 'Y';

            /* Multiply by 1/SCALE if doing so will not cause overflow */
            if (scale != ONE) {
                ix = cblas_izamax(n, work, 1);
                xnorm = fabs(creal(work[ix])) + fabs(cimag(work[ix]));
                if (scale < xnorm * smlnum || scale == ZERO) {
                    return;
                }
                zdrscl(n, scale, work, 1);
            }
        }

        /* Compute the estimate of the reciprocal condition number */
        if (ainvnm != ZERO) {
            *rcond = (ONE / anorm) / ainvnm;
        }
    }
}
