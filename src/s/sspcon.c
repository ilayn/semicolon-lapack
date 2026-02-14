/**
 * @file sspcon.c
 * @brief SSPCON estimates the reciprocal condition number of a symmetric packed matrix.
 */

#include "semicolon_lapack_single.h"

/**
 * SSPCON estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric packed matrix A using the factorization
 * A = U*D*U**T or A = L*D*L**T computed by SSPTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]     uplo   = 'U': Upper triangular, form is A = U*D*U**T
 *                        = 'L': Lower triangular, form is A = L*D*L**T
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The block diagonal matrix D and the multipliers used to
 *                       obtain the factor U or L as computed by SSPTRF, stored as a
 *                       packed triangular matrix. Array of dimension (n*(n+1)/2).
 * @param[in]     ipiv   Details of the interchanges and the block structure of D
 *                       as determined by SSPTRF. Array of dimension (n).
 * @param[in]     anorm  The 1-norm of the original matrix A.
 * @param[out]    rcond  The reciprocal of the condition number of the matrix A.
 * @param[out]    work   Workspace array of dimension (2*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sspcon(
    const char* uplo,
    const int n,
    const f32* const restrict AP,
    const int* const restrict ipiv,
    const f32 anorm,
    f32* rcond,
    f32* const restrict work,
    int* const restrict iwork,
    int* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    int upper;
    int i, ip, kase;
    f32 ainvnm;
    int isave[3];
    int info_local;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (anorm < ZERO) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("SSPCON", -(*info));
        return;
    }

    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm <= ZERO) {
        return;
    }

    if (upper) {
        ip = n * (n + 1) / 2 - 1;
        for (i = n - 1; i >= 0; i--) {
            if (ipiv[i] >= 0 && AP[ip] == ZERO)
                return;
            ip = ip - (i + 1);
        }
    } else {
        ip = 0;
        for (i = 0; i < n; i++) {
            if (ipiv[i] >= 0 && AP[ip] == ZERO)
                return;
            ip = ip + n - i;
        }
    }

    kase = 0;
    do {
        slacn2(n, &work[n], work, iwork, &ainvnm, &kase, isave);
        if (kase != 0) {
            ssptrs(uplo, n, 1, AP, ipiv, work, n, &info_local);
        }
    } while (kase != 0);

    if (ainvnm != ZERO)
        *rcond = (ONE / ainvnm) / anorm;
}
