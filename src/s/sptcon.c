/**
 * @file sptcon.c
 * @brief SPTCON computes the reciprocal of the condition number (in the
 *        1-norm) of a real symmetric positive definite tridiagonal matrix.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SPTCON computes the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric positive definite tridiagonal matrix
 * using the factorization A = L*D*L**T or A = U**T*D*U computed by
 * SPTTRF.
 *
 * Norm(inv(A)) is computed by a direct method, and the reciprocal of
 * the condition number is computed as
 *              RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]  n      The order of the matrix A. n >= 0.
 * @param[in]  D      Double precision array, dimension (n).
 *                    The n diagonal elements of the diagonal matrix D
 *                    from the factorization of A, as computed by SPTTRF.
 * @param[in]  E      Double precision array, dimension (n-1).
 *                    The (n-1) off-diagonal elements of the unit bidiagonal
 *                    factor U or L from the factorization of A, as computed
 *                    by SPTTRF.
 * @param[in]  anorm  The 1-norm of the original matrix A.
 * @param[out] rcond  The reciprocal of the condition number of the matrix A,
 *                    computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is
 *                    the 1-norm of inv(A) computed in this routine.
 * @param[out] work   Double precision array, dimension (n).
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sptcon(
    const INT n,
    const f32* restrict D,
    const f32* restrict E,
    const f32 anorm,
    f32* rcond,
    f32* restrict work,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    INT i, ix;
    f32 ainvnm;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (anorm < ZERO) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SPTCON", -(*info));
        return;
    }

    *rcond = ZERO;
    if (n == 0) {
        *rcond = ONE;
        return;
    } else if (anorm == ZERO) {
        return;
    }

    /* Check that D(0:n-1) is positive. */

    for (i = 0; i < n; i++) {
        if (D[i] <= ZERO)
            return;
    }

    /*
     * Solve M(A) * x = e, where M(A) = (m(i,j)) is given by
     *
     *    m(i,j) =  abs(A(i,j)), i = j,
     *    m(i,j) = -abs(A(i,j)), i .ne. j,
     *
     * and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**T.
     *
     * Solve M(L) * x = e.
     */

    work[0] = ONE;
    for (i = 1; i < n; i++) {
        work[i] = ONE + work[i - 1] * fabsf(E[i - 1]);
    }

    /* Solve D * M(L)**T * x = b. */

    work[n - 1] = work[n - 1] / D[n - 1];
    for (i = n - 2; i >= 0; i--) {
        work[i] = work[i] / D[i] + work[i + 1] * fabsf(E[i]);
    }

    /* Compute AINVNM = max(x(i)), 0<=i<n. */

    ix = cblas_isamax(n, work, 1);
    ainvnm = fabsf(work[ix]);

    /* Compute the reciprocal condition number. */

    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
