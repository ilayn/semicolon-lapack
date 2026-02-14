/**
 * @file dlagtf.c
 * @brief DLAGTF factorizes (T - lambda*I) = P*L*U for a tridiagonal matrix T.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DLAGTF factorizes the matrix (T - lambda*I), where T is an n by n
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
 * The parameter LAMBDA is included in the routine so that DLAGTF may
 * be used, in conjunction with DLAGTS, to obtain eigenvectors of T by
 * inverse iteration.
 *
 * @param[in]     n       The order of the matrix T. n >= 0.
 * @param[in,out] A       Double precision array, dimension (n).
 *                        On entry, the diagonal elements of T.
 *                        On exit, the n diagonal elements of U.
 * @param[in]     lambda  The scalar lambda.
 * @param[in,out] B       Double precision array, dimension (n-1).
 *                        On entry, the (n-1) super-diagonal elements of T.
 *                        On exit, the (n-1) super-diagonal elements of U.
 * @param[in,out] C       Double precision array, dimension (n-1).
 *                        On entry, the (n-1) sub-diagonal elements of T.
 *                        On exit, the (n-1) sub-diagonal elements of L.
 * @param[in]     tol     A relative tolerance used to indicate whether or
 *                        not the matrix (T - lambda*I) is nearly singular.
 *                        If tol is supplied as less than eps, where eps is the
 *                        relative machine precision, then eps is used in place
 *                        of tol.
 * @param[out]    D       Double precision array, dimension (n-2).
 *                        On exit, the (n-2) second super-diagonal elements of U.
 * @param[out]    in      Integer array, dimension (n).
 *                        On exit, contains details of the permutation matrix P.
 *                        If an interchange occurred at the k-th step of the
 *                        elimination, then in[k] = 1, otherwise in[k] = 0.
 *                        The element in[n-1] returns the smallest positive
 *                        integer j such that
 *                          |u(j,j)| <= norm((T - lambda*I)(j)) * tol,
 *                        where norm(A(j)) denotes the sum of absolute values of
 *                        the j-th row of A. If no such j exists then in[n-1]
 *                        is returned as zero. (Note: in[n-1] uses 1-based indexing.)
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value.
 */
void dlagtf(
    const int n,
    f64* restrict A,
    const f64 lambda,
    f64* restrict B,
    f64* restrict C,
    const f64 tol,
    f64* restrict D,
    int* restrict in,
    int* info)
{
    int k;
    f64 eps, mult, piv1, piv2, scale1, scale2, temp, tl;

    *info = 0;
    if (n < 0) {
        *info = -1;
        xerbla("DLAGTF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    A[0] = A[0] - lambda;
    in[n - 1] = 0;
    if (n == 1) {
        if (A[0] == 0.0) {
            in[0] = 1;
        }
        return;
    }

    eps = DBL_EPSILON;

    tl = fmax(tol, eps);
    scale1 = fabs(A[0]) + fabs(B[0]);
    for (k = 0; k < n - 1; k++) {
        A[k + 1] = A[k + 1] - lambda;
        scale2 = fabs(C[k]) + fabs(A[k + 1]);
        if (k < n - 2) {
            scale2 = scale2 + fabs(B[k + 1]);
        }
        if (A[k] == 0.0) {
            piv1 = 0.0;
        } else {
            piv1 = fabs(A[k]) / scale1;
        }
        if (C[k] == 0.0) {
            in[k] = 0;
            piv2 = 0.0;
            scale1 = scale2;
            if (k < n - 2) {
                D[k] = 0.0;
            }
        } else {
            piv2 = fabs(C[k]) / scale2;
            if (piv2 <= piv1) {
                in[k] = 0;
                scale1 = scale2;
                C[k] = C[k] / A[k];
                A[k + 1] = A[k + 1] - C[k] * B[k];
                if (k < n - 2) {
                    D[k] = 0.0;
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
        if ((fmax(piv1, piv2) <= tl) && (in[n - 1] == 0)) {
            in[n - 1] = k + 1;  /* 1-based index, as used by dlagts */
        }
    }
    if ((fabs(A[n - 1]) <= scale1 * tl) && (in[n - 1] == 0)) {
        in[n - 1] = n;  /* 1-based index */
    }
}
