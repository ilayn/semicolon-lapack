/**
 * @file slagtf.c
 * @brief SLAGTF factorizes (T - lambda*I) = P*L*U for a tridiagonal matrix T.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include "semicolon_lapack_single.h"

/**
 * SLAGTF factorizes the matrix (T - lambda*I), where T is an n by n
 * tridiagonal matrix and lambda is a scalar, as
 *
 *    T - lambda*I = P*L*U,
 *
 * where P is a permutation matrix, L is a unit lower tridiagonal matrix
 * with at most one non-zero sub-diagonal elements per column and U is
 * an upper triangular matrix with at most two non-zero super-diagonal
 * elements per column.
 *
 * The factorization is obtained by Gaussian elimination with partial
 * pivoting and implicit row scaling.
 *
 * The parameter LAMBDA is included in the routine so that SLAGTF may
 * be used, in conjunction with SLAGTS, to obtain eigenvectors of T by
 * inverse iteration.
 *
 * @param[in]     n       The order of the matrix T. n >= 0.
 * @param[in,out] A       Single precision array, dimension (n).
 *                        On entry, the diagonal elements of T.
 *                        On exit, the n diagonal elements of U.
 * @param[in]     lambda  The scalar lambda.
 * @param[in,out] B       Single precision array, dimension (n-1).
 *                        On entry, the (n-1) super-diagonal elements of T.
 *                        On exit, the (n-1) super-diagonal elements of U.
 * @param[in,out] C       Single precision array, dimension (n-1).
 *                        On entry, the (n-1) sub-diagonal elements of T.
 *                        On exit, the (n-1) sub-diagonal elements of L.
 * @param[in]     tol     A relative tolerance used to indicate whether or
 *                        not the matrix (T - lambda*I) is nearly singular.
 *                        If tol is supplied as less than eps, where eps is the
 *                        relative machine precision, then eps is used in place
 *                        of tol.
 * @param[out]    D       Single precision array, dimension (n-2).
 *                        On exit, the (n-2) second super-diagonal elements of U.
 * @param[out]    in      Integer array, dimension (n).
 *                        On exit, contains details of the permutation matrix P.
 *                        If an interchange occurred at the k-th step of the
 *                        elimination, then in[k] = 1, otherwise in[k] = 0,
 *                        for 0 <= k < n-1.
 *                        The element in[n-1] returns the smallest 0-based
 *                        index j such that
 *                          |u(j,j)| <= norm((T - lambda*I)(j)) * tol,
 *                        where norm(A(j)) denotes the sum of absolute values of
 *                        the j-th row of A. If no such j exists then in[n-1]
 *                        is returned as -1.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value.
 */
void slagtf(
    const INT n,
    f32* restrict A,
    const f32 lambda,
    f32* restrict B,
    f32* restrict C,
    const f32 tol,
    f32* restrict D,
    INT* restrict in,
    INT* info)
{
    INT k;
    f32 eps, mult, piv1, piv2, scale1, scale2, temp, tl;

    *info = 0;
    if (n < 0) {
        *info = -1;
        xerbla("SLAGTF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    A[0] = A[0] - lambda;
    in[n - 1] = -1;
    if (n == 1) {
        if (A[0] == 0.0f) {
            in[0] = 0;
        }
        return;
    }

    eps = FLT_EPSILON;

    tl = fmaxf(tol, eps);
    scale1 = fabsf(A[0]) + fabsf(B[0]);
    for (k = 0; k < n - 1; k++) {
        A[k + 1] = A[k + 1] - lambda;
        scale2 = fabsf(C[k]) + fabsf(A[k + 1]);
        if (k < n - 2) {
            scale2 = scale2 + fabsf(B[k + 1]);
        }
        if (A[k] == 0.0f) {
            piv1 = 0.0f;
        } else {
            piv1 = fabsf(A[k]) / scale1;
        }
        if (C[k] == 0.0f) {
            in[k] = 0;
            piv2 = 0.0f;
            scale1 = scale2;
            if (k < n - 2) {
                D[k] = 0.0f;
            }
        } else {
            piv2 = fabsf(C[k]) / scale2;
            if (piv2 <= piv1) {
                in[k] = 0;
                scale1 = scale2;
                C[k] = C[k] / A[k];
                A[k + 1] = A[k + 1] - C[k] * B[k];
                if (k < n - 2) {
                    D[k] = 0.0f;
                }
            } else {
                in[k] = 1;
                mult = A[k] / C[k];
                A[k] = C[k];
                temp = A[k + 1];
                A[k + 1] = B[k] - mult * temp;
                if (k < n - 2) {
                    D[k] = B[k + 1];
                    B[k + 1] = -mult * D[k];
                }
                B[k] = temp;
                C[k] = mult;
            }
        }
        if ((fmaxf(piv1, piv2) <= tl) && (in[n - 1] < 0)) {
            in[n - 1] = k;
        }
    }
    if ((fabsf(A[n - 1]) <= scale1 * tl) && (in[n - 1] < 0)) {
        in[n - 1] = n - 1;
    }
}
