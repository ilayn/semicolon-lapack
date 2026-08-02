/**
 * @file cptcon.c
 * @brief CPTCON computes the reciprocal of the condition number (in the
 *        1-norm) of a complex Hermitian positive definite tridiagonal matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPTCON computes the reciprocal of the condition number (in the
 * 1-norm) of a complex Hermitian positive definite tridiagonal matrix
 * using the factorization A = L*D*L**H or A = U**H*D*U computed by
 * CPTTRF.
 *
 * Norm(inv(A)) is computed by a direct method, and the reciprocal of
 * the condition number is computed as
 *              RCOND = 1 / (ANORM * norm(inv(A))).
 *
 * @param[in]  n      The order of the matrix A. `n>=0`.
 * @param[in]  D      Array of dimension `n`.
 *                    The n diagonal elements of the diagonal matrix D
 *                    from the factorization of A, as computed by `cpttrf`.
 * @param[in]  E      Complex array of dimension `n-1`.
 *                    The (n-1) off-diagonal elements of the unit bidiagonal
 *                    factor U or L from the factorization of A, as computed
 *                    by `cpttrf`.
 * @param[in]  anorm  The 1-norm of the original matrix A.
 * @param[out] rcond  The reciprocal of the condition number of the matrix A,
 *                    computed as `rcond=1/(anorm*ainvnm)`, where `ainvnm` is
 *                    the 1-norm of inv(A) computed in this routine.
 * @param[out] rwork  Array of dimension `n`.
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
void cptcon(
    const INT n,
    const f32* restrict D,
    const c64* restrict E,
    const f32 anorm,
    f32* rcond,
    f32* restrict rwork,
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
        xerbla("CPTCON", -(*info));
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
     * and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**H.
     *
     * Solve M(L) * x = e.
     */

    rwork[0] = ONE;
    for (i = 1; i < n; i++) {
        rwork[i] = ONE + rwork[i - 1] * cabsf(E[i - 1]);
    }

    /* Solve D * M(L)**H * x = b. */

    rwork[n - 1] = rwork[n - 1] / D[n - 1];
    for (i = n - 2; i >= 0; i--) {
        rwork[i] = rwork[i] / D[i] + rwork[i + 1] * cabsf(E[i]);
    }

    /* Compute AINVNM = max(x(i)), 0<=i<n. */

    ix = cblas_isamax(n, rwork, 1);
    ainvnm = fabsf(rwork[ix]);

    /* Compute the reciprocal condition number. */

    if (ainvnm != ZERO) {
        *rcond = (ONE / ainvnm) / anorm;
    }
}
