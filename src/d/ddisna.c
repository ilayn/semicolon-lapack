/**
 * @file ddisna.c
 * @brief DDISNA computes the reciprocal condition numbers for the eigenvectors.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DDISNA computes the reciprocal condition numbers for the eigenvectors
 * of a real symmetric or complex Hermitian matrix or for the left or
 * right singular vectors of a general m-by-n matrix. The reciprocal
 * condition number is the 'gap' between the corresponding eigenvalue or
 * singular value and the nearest other one.
 *
 * The bound on the error, measured by angle in radians, in the I-th
 * computed vector is given by
 *
 *        DLAMCH( 'E' ) * ( ANORM / SEP( I ) )
 *
 * where ANORM = 2-norm(A) = max( abs( D(j) ) ).  SEP(I) is not allowed
 * to be smaller than DLAMCH( 'E' )*ANORM in order to limit the size of
 * the error bound.
 *
 * DDISNA may also be used to compute error bounds for eigenvectors of
 * the generalized symmetric definite eigenproblem.
 *
 * @param[in] job
 *          Specifies for which problem the reciprocal condition numbers
 *          should be computed:
 *          = 'E': the eigenvectors of a symmetric/Hermitian matrix;
 *          = 'L': the left singular vectors of a general matrix;
 *          = 'R': the right singular vectors of a general matrix.
 *
 * @param[in] m
 *          The number of rows of the matrix. m >= 0.
 *
 * @param[in] n
 *          If job = 'L' or 'R', the number of columns of the matrix,
 *          in which case n >= 0. Ignored if job = 'E'.
 *
 * @param[in] D
 *          Double precision array, dimension (m) if job = 'E'
 *                          dimension (min(m,n)) if job = 'L' or 'R'
 *          The eigenvalues (if job = 'E') or singular values (if job =
 *          'L' or 'R') of the matrix, in either increasing or decreasing
 *          order. If singular values, they must be non-negative.
 *
 * @param[out] SEP
 *          Double precision array, dimension (m) if job = 'E'
 *                           dimension (min(m,n)) if job = 'L' or 'R'
 *          The reciprocal condition numbers of the vectors.
 *
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ddisna(
    const char* job,
    const INT m,
    const INT n,
    const f64* restrict D,
    f64* restrict SEP,
    INT* info)
{
    const f64 zero = 0.0;

    INT eigen, left, right, sing;
    INT i, k;
    INT incr, decr;
    f64 anorm, eps, newgap, oldgap, safmin, thresh;

    *info = 0;
    eigen = (job[0] == 'E' || job[0] == 'e');
    left = (job[0] == 'L' || job[0] == 'l');
    right = (job[0] == 'R' || job[0] == 'r');
    sing = left || right;

    if (eigen) {
        k = m;
    } else if (sing) {
        k = (m < n) ? m : n;
    } else {
        k = 0;
    }

    if (!eigen && !sing) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (k < 0) {
        *info = -3;
    } else {
        incr = 1;
        decr = 1;
        for (i = 0; i < k - 1; i++) {
            if (incr) {
                incr = (D[i] <= D[i + 1]);
            }
            if (decr) {
                decr = (D[i] >= D[i + 1]);
            }
        }
        if (sing && k > 0) {
            if (incr) {
                incr = (zero <= D[0]);
            }
            if (decr) {
                decr = (D[k - 1] >= zero);
            }
        }
        if (!(incr || decr)) {
            *info = -4;
        }
    }
    if (*info != 0) {
        xerbla("DDISNA", -(*info));
        return;
    }

    if (k == 0) {
        return;
    }

    if (k == 1) {
        SEP[0] = dlamch("O");
    } else {
        oldgap = fabs(D[1] - D[0]);
        SEP[0] = oldgap;
        for (i = 1; i < k - 1; i++) {
            newgap = fabs(D[i + 1] - D[i]);
            SEP[i] = (oldgap < newgap) ? oldgap : newgap;
            oldgap = newgap;
        }
        SEP[k - 1] = oldgap;
    }
    if (sing) {
        if ((left && m > n) || (right && m < n)) {
            if (incr) {
                SEP[0] = (SEP[0] < D[0]) ? SEP[0] : D[0];
            }
            if (decr) {
                SEP[k - 1] = (SEP[k - 1] < D[k - 1]) ? SEP[k - 1] : D[k - 1];
            }
        }
    }

    eps = dlamch("E");
    safmin = dlamch("S");
    anorm = fabs(D[0]);
    if (fabs(D[k - 1]) > anorm) {
        anorm = fabs(D[k - 1]);
    }
    if (anorm == zero) {
        thresh = eps;
    } else {
        thresh = eps * anorm;
        if (safmin > thresh) thresh = safmin;
    }
    for (i = 0; i < k; i++) {
        if (SEP[i] < thresh) SEP[i] = thresh;
    }
}
