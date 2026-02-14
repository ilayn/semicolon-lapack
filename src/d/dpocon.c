/**
 * @file dpocon.c
 * @brief DPOCON estimates the reciprocal of the condition number of a
 *        symmetric positive definite matrix.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPOCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite matrix using the
 * Cholesky factorization A = U**T*U or A = L*L**T computed by DPOTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, as computed
 *                       by dpotrf. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the symmetric matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM).
 * @param[out]    work   Double precision array, dimension (3*n).
 * @param[out]    iwork  Integer array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void dpocon(
    const char* uplo,
    const int n,
    const f64* restrict A,
    const int lda,
    const f64 anorm,
    f64* rcond,
    f64* restrict work,
    int* restrict iwork,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (anorm < ZERO) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("DPOCON", -(*info));
        return;
    }

    // Quick return if possible
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    // Safe minimum
    f64 smlnum = dlamch("S");

    // Estimate the 1-norm of inv(A).
    int kase = 0;
    char normin = 'N';
    int isave[3] = {0, 0, 0};
    f64 ainvnm;

    for (;;) {
        dlacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase == 0) break;

        f64 scalel, scaleu;
        int linfo;

        if (upper) {
            // Multiply by inv(U**T).
            dlatrs("U", "T", "N", &normin, n, A, lda, work, &scalel,
                   &work[2 * n], &linfo);
            normin = 'Y';

            // Multiply by inv(U).
            dlatrs("U", "N", "N", &normin, n, A, lda, work, &scaleu,
                   &work[2 * n], &linfo);
        } else {
            // Multiply by inv(L).
            dlatrs("L", "N", "N", &normin, n, A, lda, work, &scalel,
                   &work[2 * n], &linfo);
            normin = 'Y';

            // Multiply by inv(L**T).
            dlatrs("L", "T", "N", &normin, n, A, lda, work, &scaleu,
                   &work[2 * n], &linfo);
        }

        // Multiply by 1/SCALE if doing so will not cause overflow.
        f64 scale = scalel * scaleu;
        if (scale != ONE) {
            int ix = cblas_idamax(n, work, 1);
            if (scale < fabs(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            drscl(n, scale, work, 1);
        }
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
