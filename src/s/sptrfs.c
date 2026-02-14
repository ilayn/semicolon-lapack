/**
 * @file sptrfs.c
 * @brief SPTRFS improves the computed solution to a system of linear equations
 *        when the coefficient matrix is symmetric positive definite and
 *        tridiagonal, and provides error bounds and backward error estimates.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPTRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric positive definite
 * and tridiagonal, and provides error bounds and backward error
 * estimates for the solution.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number
 *                      of columns of the matrix B. nrhs >= 0.
 * @param[in]     D     Double precision array, dimension (n).
 *                      The n diagonal elements of the tridiagonal matrix A.
 * @param[in]     E     Double precision array, dimension (n-1).
 *                      The (n-1) subdiagonal elements of the tridiagonal matrix A.
 * @param[in]     DF    Double precision array, dimension (n).
 *                      The n diagonal elements of the diagonal matrix D from
 *                      the factorization computed by SPTTRF.
 * @param[in]     EF    Double precision array, dimension (n-1).
 *                      The (n-1) subdiagonal elements of the unit bidiagonal
 *                      factor L from the factorization computed by SPTTRF.
 * @param[in]     B     Double precision array, dimension (ldb, nrhs).
 *                      The right hand side matrix B.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[in,out] X     Double precision array, dimension (ldx, nrhs).
 *                      On entry, the solution matrix X, as computed by SPTTRS.
 *                      On exit, the improved solution matrix X.
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1,n).
 * @param[out]    ferr  Double precision array, dimension (nrhs).
 *                      The forward error bound for each solution vector X(j).
 * @param[out]    berr  Double precision array, dimension (nrhs).
 *                      The componentwise relative backward error of each
 *                      solution vector X(j).
 * @param[out]    work  Double precision array, dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sptrfs(
    const int n,
    const int nrhs,
    const f32* restrict D,
    const f32* restrict E,
    const f32* restrict DF,
    const f32* restrict EF,
    const f32* restrict B,
    const int ldb,
    f32* restrict X,
    const int ldx,
    f32* restrict ferr,
    f32* restrict berr,
    f32* restrict work,
    int* info)
{
    const int ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    int count, i, ix, j, nz;
    f32 bi, cx, dx, ex, eps, lstres, s, safe1, safe2, safmin;
    int max_n_1 = (1 > n) ? 1 : n;
    int info_local;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (ldb < max_n_1) {
        *info = -8;
    } else if (ldx < max_n_1) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("SPTRFS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */

    nz = 4;
    eps = slamch("Epsilon");
    safmin = slamch("Safe minimum");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    /* Do for each right hand side */

    for (j = 0; j < nrhs; j++) {

        count = 1;
        lstres = THREE;

        /*
         * Loop until stopping criterion is satisfied.
         *
         * Compute residual R = B - A * X.  Also compute
         * abs(A)*abs(x) + abs(b) for use in the backward error bound.
         */
refine:
        if (n == 1) {
            bi = B[0 + j * ldb];
            dx = D[0] * X[0 + j * ldx];
            work[n] = bi - dx;
            work[0] = fabsf(bi) + fabsf(dx);
        } else {
            bi = B[0 + j * ldb];
            dx = D[0] * X[0 + j * ldx];
            ex = E[0] * X[1 + j * ldx];
            work[n] = bi - dx - ex;
            work[0] = fabsf(bi) + fabsf(dx) + fabsf(ex);
            for (i = 1; i < n - 1; i++) {
                bi = B[i + j * ldb];
                cx = E[i - 1] * X[(i - 1) + j * ldx];
                dx = D[i] * X[i + j * ldx];
                ex = E[i] * X[(i + 1) + j * ldx];
                work[n + i] = bi - cx - dx - ex;
                work[i] = fabsf(bi) + fabsf(cx) + fabsf(dx) + fabsf(ex);
            }
            bi = B[(n - 1) + j * ldb];
            cx = E[n - 2] * X[(n - 2) + j * ldx];
            dx = D[n - 1] * X[(n - 1) + j * ldx];
            work[n + n - 1] = bi - cx - dx;
            work[n - 1] = fabsf(bi) + fabsf(cx) + fabsf(dx);
        }

        /*
         * Compute componentwise relative backward error from formula
         *
         * max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
         *
         * where abs(Z) is the componentwise absolute value of the matrix
         * or vector Z.  If the i-th component of the denominator is less
         * than SAFE2, then SAFE1 is added to the i-th components of the
         * numerator and denominator before dividing.
         */

        s = ZERO;
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                s = (s > fabsf(work[n + i]) / work[i]) ? s : fabsf(work[n + i]) / work[i];
            } else {
                s = (s > (fabsf(work[n + i]) + safe1) / (work[i] + safe1))
                    ? s : (fabsf(work[n + i]) + safe1) / (work[i] + safe1);
            }
        }
        berr[j] = s;

        /*
         * Test stopping criterion. Continue iterating if
         *   1) The residual BERR(J) is larger than machine epsilon, and
         *   2) BERR(J) decreased by at least a factor of 2 during the
         *      last iteration, and
         *   3) At most ITMAX iterations tried.
         */

        if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {

            /* Update solution and try again. */

            spttrs(n, 1, DF, EF, &work[n], n, &info_local);
            cblas_saxpy(n, ONE, &work[n], 1, &X[0 + j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
            goto refine;
        }

        /*
         * Bound error from formula
         *
         * norm(X - XTRUE) / norm(X) .le. FERR =
         * norm( abs(inv(A))*
         *    ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X)
         *
         * where
         *   norm(Z) is the magnitude of the largest component of Z
         *   inv(A) is the inverse of A
         *   abs(Z) is the componentwise absolute value of the matrix or
         *      vector Z
         *   NZ is the maximum number of nonzeros in any row of A, plus 1
         *   EPS is machine epsilon
         *
         * The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B))
         * is incremented by SAFE1 if the i-th component of
         * abs(A)*abs(X) + abs(B) is less than SAFE2.
         */

        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }
        ix = cblas_isamax(n, work, 1);
        ferr[j] = work[ix];

        /*
         * Estimate the norm of inv(A).
         *
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
            work[i] = ONE + work[i - 1] * fabsf(EF[i - 1]);
        }

        /* Solve D * M(L)**T * x = b. */

        work[n - 1] = work[n - 1] / DF[n - 1];
        for (i = n - 2; i >= 0; i--) {
            work[i] = work[i] / DF[i] + work[i + 1] * fabsf(EF[i]);
        }

        /* Compute norm(inv(A)) = max(x(i)), 0<=i<n. */

        ix = cblas_isamax(n, work, 1);
        ferr[j] = ferr[j] * fabsf(work[ix]);

        /* Normalize error. */

        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > fabsf(X[i + j * ldx])) ? lstres : fabsf(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
