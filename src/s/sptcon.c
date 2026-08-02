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
 * @param[in]  n      The order of the matrix A. `n>=0`.
 * @param[in]  D      Array of dimension `n`.
 *                    The n diagonal elements of the diagonal matrix D
 *                    from the factorization of A, as computed by `spttrf`.
 * @param[in]  E      Array of dimension `n-1`.
 *                    The (n-1) off-diagonal elements of the unit bidiagonal
 *                    factor U or L from the factorization of A, as computed
 *                    by `spttrf`.
 * @param[in]  anorm  The 1-norm of the original matrix A.
 * @param[out] rcond  The reciprocal of the condition number of the matrix A,
 *                    computed as `rcond=1/(anorm*ainvnm)`, where `ainvnm` is
 *                    the 1-norm of inv(A) computed in this routine.
 * @param[out] work   Array of dimension `n`.
 * @param[out] info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *
 * @par Further Details:
 * @rst
 * The method used is described in Nicholas J. Higham, "Efficient
 * Algorithms for Computing the Condition Number of a Tridiagonal
 * Matrix", SIAM J. Sci. Stat. Comput., Vol. 7, No. 1, January 1986.
 * @endrst
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
