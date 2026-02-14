#include "semicolon_lapack_complex_double.h"
#include <complex.h>
/**
 * @file zgtcon.c
 * @brief ZGTCON estimates the reciprocal of the condition number of a
 *        tridiagonal matrix using the LU factorization.
 */

/**
 * ZGTCON estimates the reciprocal of the condition number of a complex
 * tridiagonal matrix A using the LU factorization as computed by
 * ZGTTRF.
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
 *                    LU factorization of A as computed by ZGTTRF.
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
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zgtcon(
    const char* norm,
    const int n,
    const c128* const restrict DL,
    const c128* const restrict D,
    const c128* const restrict DU,
    const c128* const restrict DU2,
    const int* const restrict ipiv,
    const f64 anorm,
    f64* rcond,
    c128* const restrict work,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int onenrm;
    int i, kase, kase1;
    f64 ainvnm;
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
        xerbla("ZGTCON", -(*info));
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
        if (D[i] == CMPLX(0.0, 0.0)) {
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
        zlacn2(n, work + n, work, &ainvnm, &kase, isave);

        if (kase == 0) {
            break;
        }

        if (kase == kase1) {
            /* Multiply by inv(U)*inv(L) */
            zgttrs("N", n, 1, DL, D, DU, DU2, ipiv, work, ldb, info);
        } else {
            /* Multiply by inv(L**H)*inv(U**H) */
            zgttrs("C", n, 1, DL, D, DU, DU2, ipiv, work, ldb, info);
        }
    }

    /* Compute the estimate of the reciprocal condition number */
    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
