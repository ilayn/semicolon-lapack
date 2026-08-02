/**
 * @file cppcon.c
 * @brief CPPCON estimates the reciprocal of the condition number of a Hermitian positive definite packed matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPPCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite packed matrix using
 * the Cholesky factorization A = U**H*U or A = L*L**H computed by
 * CPPTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     AP     Array of dimension `n*(n+1)/2`.
 *                       The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, packed
 *                       columnwise in a linear array. The j-th column of
 *                       U or L is stored in the array AP as follows:
 *                       if `uplo='U'`, `AP[i + j*(j+1)/2] = U(i,j)` for `0<=i<=j`;
 *                       if `uplo='L'`, `AP[i + j*(2*n-j-1)/2] = L(i,j)` for
 *                       `j<=i<=n-1`.
 * @param[in]     anorm  The 1-norm (or infinity-norm) of the Hermitian matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A,
 *                       computed as `rcond=1/(anorm*ainvnm)`, where `ainvnm` is
 *                       an estimate of the 1-norm of inv(A) computed in this
 *                       routine.
 * @param[out]    work   Complex array of dimension `2*n`.
 * @param[out]    rwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void cppcon(
    const char* uplo,
    const INT n,
    const c64* restrict AP,
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
    } else if (anorm < ZERO) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CPPCON", -(*info));
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

    while (1) {
        clacn2(n, &work[n], work, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (upper) {
            clatps("U", "C", "N", &normin, n, AP, work, &scalel, rwork, info);
            normin = 'Y';

            clatps("U", "N", "N", &normin, n, AP, work, &scaleu, rwork, info);
        } else {
            clatps("L", "N", "N", &normin, n, AP, work, &scalel, rwork, info);
            normin = 'Y';

            clatps("L", "C", "N", &normin, n, AP, work, &scaleu, rwork, info);
        }

        scale = scalel * scaleu;
        if (scale != ONE) {
            ix = cblas_icamax(n, work, 1);
            if (scale < cabs1f(work[ix]) * smlnum || scale == ZERO) {
                return;
            }
            csrscl(n, scale, work, 1);
        }
    }

    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
