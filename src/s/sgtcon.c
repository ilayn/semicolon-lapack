#include "semicolon_lapack_single.h"
/**
 * @file sgtcon.c
 * @brief SGTCON estimates the reciprocal of the condition number of a
 *        tridiagonal matrix using the LU factorization.
 */

/**
 * SGTCON estimates the reciprocal of the condition number of a real
 * tridiagonal matrix A using the LU factorization as computed by
 * SGTTRF.
 *
 * An estimate is obtained for norm(inv(A)), and the reciprocal of the
 * condition number is computed as RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]  norm   Specifies whether the 1-norm condition number or the
 *                    infinity-norm condition number is required:
 *                    = '1' or 'O': 1-norm
 *                    = 'I': Infinity-norm
 * @param[in]  n      The order of the matrix A. n >= 0.
 * @param[in]  DL     The (n-1) multipliers that define the matrix L from the
 *                    LU factorization of A as computed by SGTTRF.
 *                    Array of dimension (n-1).
 * @param[in]  D      The n diagonal elements of the upper triangular matrix U
 *                    from the LU factorization of A. Array of dimension (n).
 * @param[in]  DU     The (n-1) elements of the first superdiagonal of U.
 *                    Array of dimension (n-1).
 * @param[in]  DU2    The (n-2) elements of the second superdiagonal of U.
 *                    Array of dimension (n-2).
 * @param[in]  ipiv   The pivot indices; for 0 <= i < n, row i of the matrix was
 *                    interchanged with row ipiv[i]. Array of dimension (n).
 * @param[in]  anorm  If norm = '1' or "O", the 1-norm of the original matrix A.
 *                    If norm = "I", the infinity-norm of the original matrix A.
 * @param[out] rcond  The reciprocal of the condition number of the matrix A,
 *                    computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is an
 *                    estimate of the 1-norm of inv(A) computed in this routine.
 * @param[out] work   Workspace array of dimension (2*n).
 * @param[out] iwork  Integer workspace array of dimension (n).
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sgtcon(
    const char* norm,
    const int n,
    const f32* restrict DL,
    const f32* restrict D,
    const f32* restrict DU,
    const f32* restrict DU2,
    const int* restrict ipiv,
    const f32 anorm,
    f32* rcond,
    f32* restrict work,
    int* restrict iwork,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int onenrm;
    int i, kase, kase1;
    f32 ainvnm;
    int isave[3];
    int ldb;

    /* Test the input arguments */
    *info = 0;
    onenrm = (norm[0] == '1' || norm[0] == 'O' || norm[0] == 'o');

    if (!onenrm && !(norm[0] == 'I' || norm[0] == 'i')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (anorm < ZERO) {
        *info = -8;
    }

    if (*info != 0) {
        xerbla("SGTCON", -(*info));
        return;
    }

    /* Quick return if possible */
    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    /* Check that D[0:n-1] is non-zero */
    for (i = 0; i < n; i++) {
        if (D[i] == ZERO) {
            return;
        }
    }

    ainvnm = ZERO;
    if (onenrm) {
        kase1 = 1;
    } else {
        kase1 = 2;
    }

    kase = 0;
    ldb = (n > 1) ? n : 1;

    /* Reverse communication loop for norm estimation */
    for (;;) {
        slacn2(n, work + n, work, iwork, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (kase == kase1) {
            /* Multiply by inv(U)*inv(L) */
            sgttrs("N", n, 1, DL, D, DU, DU2, ipiv, work, ldb, info);
        } else {
            /* Multiply by inv(L**T)*inv(U**T) */
            sgttrs("T", n, 1, DL, D, DU, DU2, ipiv, work, ldb, info);
        }
    }

    /* Compute the estimate of the reciprocal condition number */
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
