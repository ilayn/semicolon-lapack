/**
 * @file zpocon.c
 * @brief ZPOCON estimates the reciprocal of the condition number of a
 *        complex Hermitian positive definite matrix.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZPOCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite matrix using the
 * Cholesky factorization A = U**H*U or A = L*L**H computed by ZPOTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, as computed
 *                       by zpotrf. Complex array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the Hermitian matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM).
 * @param[out]    work   Complex array, dimension (2*n).
 * @param[out]    rwork  Double precision array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void zpocon(
    const char* uplo,
    const INT n,
    const c128* restrict A,
    const INT lda,
    const f64 anorm,
    f64* rcond,
    c128* restrict work,
    f64* restrict rwork,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
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
        xerbla("ZPOCON", -(*info));
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

    f64 smlnum = dlamch("S");

    // Estimate the 1-norm of inv(A).
    INT kase = 0;
    char normin = 'N';
    INT isave[3] = {0, 0, 0};
    f64 ainvnm;

    for (;;) {
        zlacn2(n, &work[n], work, &ainvnm, &kase, isave);
        if (kase == 0) break;

        f64 scalel, scaleu;
        INT linfo;

        if (upper) {
            // Multiply by inv(U**H).
            zlatrs("U", "C", "N", &normin, n, A, lda, work, &scalel,
                   rwork, &linfo);
            normin = 'Y';

            // Multiply by inv(U).
            zlatrs("U", "N", "N", &normin, n, A, lda, work, &scaleu,
                   rwork, &linfo);
        } else {
            // Multiply by inv(L).
            zlatrs("L", "N", "N", &normin, n, A, lda, work, &scalel,
                   rwork, &linfo);
            normin = 'Y';

            // Multiply by inv(L**H).
            zlatrs("L", "C", "N", &normin, n, A, lda, work, &scaleu,
                   rwork, &linfo);
        }

        // Multiply by 1/SCALE if doing so will not cause overflow.
        f64 scale = scalel * scaleu;
        if (scale != ONE) {
            INT ix = cblas_izamax(n, work, 1);
            if (scale < cabs1(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            zdrscl(n, scale, work, 1);
        }
    }

    // Compute the estimate of the reciprocal condition number.
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
