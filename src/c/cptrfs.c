/**
 * @file cptrfs.c
 * @brief CPTRFS improves the computed solution to a system of linear equations
 *        when the coefficient matrix is Hermitian positive definite and
 *        tridiagonal, and provides error bounds and backward error estimates.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPTRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian positive definite
 * and tridiagonal, and provides error bounds and backward error
 * estimates for the solution.
 *
 * @param[in]     uplo  Specifies whether the superdiagonal or the subdiagonal
 *                      of the tridiagonal matrix A is stored and the form of
 *                      the factorization:
 *                      = 'U':  E is the superdiagonal of A, and A = U**H*D*U;
 *                      = 'L':  E is the subdiagonal of A, and A = L*D*L**H.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number
 *                      of columns of the matrix B. nrhs >= 0.
 * @param[in]     D     Single precision array, dimension (n).
 *                      The n real diagonal elements of the tridiagonal matrix A.
 * @param[in]     E     Single complex array, dimension (n-1).
 *                      The (n-1) off-diagonal elements of the tridiagonal
 *                      matrix A (see UPLO).
 * @param[in]     DF    Single precision array, dimension (n).
 *                      The n diagonal elements of the diagonal matrix D from
 *                      the factorization computed by CPTTRF.
 * @param[in]     EF    Single complex array, dimension (n-1).
 *                      The (n-1) off-diagonal elements of the unit bidiagonal
 *                      factor U or L from the factorization computed by CPTTRF
 *                      (see UPLO).
 * @param[in]     B     Single complex array, dimension (ldb, nrhs).
 *                      The right hand side matrix B.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[in,out] X     Single complex array, dimension (ldx, nrhs).
 *                      On entry, the solution matrix X, as computed by CPTTRS.
 *                      On exit, the improved solution matrix X.
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1,n).
 * @param[out]    ferr  Single precision array, dimension (nrhs).
 *                      The forward error bound for each solution vector X(j).
 * @param[out]    berr  Single precision array, dimension (nrhs).
 *                      The componentwise relative backward error of each
 *                      solution vector X(j).
 * @param[out]    work  Single complex array, dimension (n).
 * @param[out]    rwork Single precision array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cptrfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f32* restrict D,
    const c64* restrict E,
    const f32* restrict DF,
    const c64* restrict EF,
    const c64* restrict B,
    const INT ldb,
    c64* restrict X,
    const INT ldx,
    f32* restrict ferr,
    f32* restrict berr,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const INT ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT upper;
    INT count, i, ix, j, nz;
    f32 eps, lstres, s, safe1, safe2, safmin;
    c64 bi, cx, dx, ex;
    INT max_n_1 = (1 > n) ? 1 : n;
    INT info_local;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < max_n_1) {
        *info = -9;
    } else if (ldx < max_n_1) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("CPTRFS", -(*info));
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
        if (upper) {
            if (n == 1) {
                bi = B[0 + j * ldb];
                dx = D[0] * X[0 + j * ldx];
                work[0] = bi - dx;
                rwork[0] = cabs1f(bi) + cabs1f(dx);
            } else {
                bi = B[0 + j * ldb];
                dx = D[0] * X[0 + j * ldx];
                ex = E[0] * X[1 + j * ldx];
                work[0] = bi - dx - ex;
                rwork[0] = cabs1f(bi) + cabs1f(dx) +
                           cabs1f(E[0]) * cabs1f(X[1 + j * ldx]);
                for (i = 1; i < n - 1; i++) {
                    bi = B[i + j * ldb];
                    cx = conjf(E[i - 1]) * X[(i - 1) + j * ldx];
                    dx = D[i] * X[i + j * ldx];
                    ex = E[i] * X[(i + 1) + j * ldx];
                    work[i] = bi - cx - dx - ex;
                    rwork[i] = cabs1f(bi) +
                               cabs1f(E[i - 1]) * cabs1f(X[(i - 1) + j * ldx]) +
                               cabs1f(dx) + cabs1f(E[i]) *
                               cabs1f(X[(i + 1) + j * ldx]);
                }
                bi = B[(n - 1) + j * ldb];
                cx = conjf(E[n - 2]) * X[(n - 2) + j * ldx];
                dx = D[n - 1] * X[(n - 1) + j * ldx];
                work[n - 1] = bi - cx - dx;
                rwork[n - 1] = cabs1f(bi) + cabs1f(E[n - 2]) *
                               cabs1f(X[(n - 2) + j * ldx]) + cabs1f(dx);
            }
        } else {
            if (n == 1) {
                bi = B[0 + j * ldb];
                dx = D[0] * X[0 + j * ldx];
                work[0] = bi - dx;
                rwork[0] = cabs1f(bi) + cabs1f(dx);
            } else {
                bi = B[0 + j * ldb];
                dx = D[0] * X[0 + j * ldx];
                ex = conjf(E[0]) * X[1 + j * ldx];
                work[0] = bi - dx - ex;
                rwork[0] = cabs1f(bi) + cabs1f(dx) +
                           cabs1f(E[0]) * cabs1f(X[1 + j * ldx]);
                for (i = 1; i < n - 1; i++) {
                    bi = B[i + j * ldb];
                    cx = E[i - 1] * X[(i - 1) + j * ldx];
                    dx = D[i] * X[i + j * ldx];
                    ex = conjf(E[i]) * X[(i + 1) + j * ldx];
                    work[i] = bi - cx - dx - ex;
                    rwork[i] = cabs1f(bi) +
                               cabs1f(E[i - 1]) * cabs1f(X[(i - 1) + j * ldx]) +
                               cabs1f(dx) + cabs1f(E[i]) *
                               cabs1f(X[(i + 1) + j * ldx]);
                }
                bi = B[(n - 1) + j * ldb];
                cx = E[n - 2] * X[(n - 2) + j * ldx];
                dx = D[n - 1] * X[(n - 1) + j * ldx];
                work[n - 1] = bi - cx - dx;
                rwork[n - 1] = cabs1f(bi) + cabs1f(E[n - 2]) *
                               cabs1f(X[(n - 2) + j * ldx]) + cabs1f(dx);
            }
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
            if (rwork[i] > safe2) {
                s = (s > cabs1f(work[i]) / rwork[i]) ? s : cabs1f(work[i]) / rwork[i];
            } else {
                s = (s > (cabs1f(work[i]) + safe1) / (rwork[i] + safe1))
                    ? s : (cabs1f(work[i]) + safe1) / (rwork[i] + safe1);
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

            cpttrs(uplo, n, 1, DF, EF, work, n, &info_local);
            const c64 CONE = CMPLXF(ONE, 0.0f);
            cblas_caxpy(n, &CONE, work, 1, &X[0 + j * ldx], 1);
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
            if (rwork[i] > safe2) {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }
        ix = cblas_isamax(n, rwork, 1);
        ferr[j] = rwork[ix];

        /*
         * Estimate the norm of inv(A).
         *
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
            rwork[i] = ONE + rwork[i - 1] * cabsf(EF[i - 1]);
        }

        /* Solve D * M(L)**H * x = b. */

        rwork[n - 1] = rwork[n - 1] / DF[n - 1];
        for (i = n - 2; i >= 0; i--) {
            rwork[i] = rwork[i] / DF[i] + rwork[i + 1] * cabsf(EF[i]);
        }

        /* Compute norm(inv(A)) = max(x(i)), 0<=i<n. */

        ix = cblas_isamax(n, rwork, 1);
        ferr[j] = ferr[j] * fabsf(rwork[ix]);

        /* Normalize error. */

        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > cabsf(X[i + j * ldx])) ? lstres : cabsf(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
