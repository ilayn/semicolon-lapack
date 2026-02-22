/**
 * @file zppcon.c
 * @brief ZPPCON estimates the reciprocal of the condition number of a Hermitian positive definite packed matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite packed matrix using
 * the Cholesky factorization A = U**H*U or A = L*L**H computed by
 * ZPPTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, packed
 *                       columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the Hermitian matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as RCOND = 1/(ANORM * AINVNM).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Double precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zppcon(
    const char* uplo,
    const INT n,
    const c128* restrict AP,
    const f64 anorm,
    f64* rcond,
    c128* restrict work,
    f64* restrict rwork,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT upper;
    char normin;
    INT ix, kase;
    f64 ainvnm, scale, scalel, scaleu, smlnum;
    INT isave[3];

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
        xerbla("ZPPCON", -(*info));
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

    while (1) {
        zlacn2(n, &work[n], work, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (upper) {
            zlatps("U", "C", "N", &normin, n, AP, work, &scalel, rwork, info);
            normin = 'Y';

            zlatps("U", "N", "N", &normin, n, AP, work, &scaleu, rwork, info);
        } else {
            zlatps("L", "N", "N", &normin, n, AP, work, &scalel, rwork, info);
            normin = 'Y';

            zlatps("L", "C", "N", &normin, n, AP, work, &scaleu, rwork, info);
        }

        scale = scalel * scaleu;
        if (scale != ONE) {
            ix = cblas_izamax(n, work, 1);
            if (scale < cabs1(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            zdrscl(n, scale, work, 1);
        }
    }

    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
