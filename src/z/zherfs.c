/**
 * @file zherfs.c
 * @brief ZHERFS improves the computed solution to a system of linear
 *        equations with a Hermitian indefinite matrix and provides
 *        error bounds and backward error estimates.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZHERFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian indefinite, and
 * provides error bounds and backward error estimates for the solution.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangle of A is stored
 *                      - `'L'`: Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides. `nrhs>=0`.
 * @param[in]     A     Double complex array of dimension `(lda,n)`.
 *                      The Hermitian matrix A. If `uplo='U'`, the leading
 *                      N-by-N upper triangular part contains the upper
 *                      triangular part of A and the strictly lower
 *                      triangular part is not referenced. If `uplo='L'`,
 *                      the leading N-by-N lower triangular part contains
 *                      the lower triangular part of A and the strictly
 *                      upper triangular part is not referenced.
 * @param[in]     lda   The leading dimension of A. `lda>=max(1,n)`.
 * @param[in]     AF    Double complex array of dimension `(ldaf,n)`.
 *                      The factored form of A. AF contains
 *                      the block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L from the factorization
 *                      A = U*D*U**H or A = L*D*L**H as computed by `zhetrf`.
 * @param[in]     ldaf  The leading dimension of AF. `ldaf>=max(1,n)`.
 * @param[in]     ipiv  Integer array of dimension `n`. Details of the
 *                      interchanges and block structure of D from `zhetrf`.
 * @param[in]     B     Double complex array of dimension `(ldb,nrhs)`.
 *                      The right hand side matrix B.
 * @param[in]     ldb   The leading dimension of B. `ldb>=max(1,n)`.
 * @param[in,out] X     Double complex array of dimension `(ldx,nrhs)`.
 *                      On entry, the solution matrix X, as computed by `zhetrs`.
 *                      On exit, the improved solution matrix X.
 * @param[in]     ldx   The leading dimension of X. `ldx>=max(1,n)`.
 * @param[out]    ferr  Real array of dimension `nrhs`. The estimated forward
 *                      error bound for each solution vector X(j). If XTRUE
 *                      is the true solution corresponding to X(j), `ferr[j]`
 *                      estimates the largest element of X(j)-XTRUE divided
 *                      by the largest element of X(j).
 * @param[out]    berr  Real array of dimension `nrhs`. The componentwise
 *                      relative backward error of each solution vector X(j),
 *                      i.e., the smallest relative change in an element of
 *                      A or B that makes X(j) an exact solution.
 * @param[out]    work  Double complex workspace array of dimension `2*n`.
 * @param[out]    rwork Real workspace array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an
 *                           illegal value
 */
void zherfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* restrict A,
    const INT lda,
    const c128* restrict AF,
    const INT ldaf,
    const INT* restrict ipiv,
    const c128* restrict B,
    const INT ldb,
    c128* restrict X,
    const INT ldx,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    f64* restrict rwork,
    INT* info)
{
    const INT ITMAX = 5;
    const f64 ZERO = 0.0;
    const c128 ONE = CMPLX(1.0, 0.0);
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    INT upper;
    INT count, i, j, k, kase, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT linfo;
    c128 neg_one = CMPLX(-1.0, 0.0);

    /* Test the input parameters. */
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        xerbla("ZHERFS", -(*info));
        return;
    }

    /* Quick return if possible. */
    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    /* NZ = maximum number of nonzero elements in each row of A, plus 1. */
    nz = n + 1;
    eps = DBL_EPSILON;
    safmin = DBL_MIN;
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    /* Do for each right hand side. */
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        /* Iterative refinement loop. */
        while (1) {
            /* Compute residual R = B - A * X. */
            cblas_zcopy(n, &B[j * ldb], 1, work, 1);
            cblas_zhemv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, &neg_one, A, lda, &X[j * ldx], 1, &ONE, work, 1);

            /* Compute componentwise relative backward error from formula
             *
             *   max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
             *
             * where abs(Z) is the componentwise absolute value of the matrix
             * or vector Z.  If the i-th component of the denominator is less
             * than SAFE2, then SAFE1 is added to the i-th components of the
             * numerator and denominator before dividing. */

            for (i = 0; i < n; i++) {
                rwork[i] = cabs1(B[i + j * ldb]);
            }

            /* Compute abs(A)*abs(X) + abs(B). */
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1(X[k + j * ldx]);
                    for (i = 0; i < k; i++) {
                        rwork[i] = rwork[i] + cabs1(A[i + k * lda]) * xk;
                        s = s + cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + fabs(creal(A[k + k * lda])) * xk + s;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1(X[k + j * ldx]);
                    rwork[k] = rwork[k] + fabs(creal(A[k + k * lda])) * xk;
                    for (i = k + 1; i < n; i++) {
                        rwork[i] = rwork[i] + cabs1(A[i + k * lda]) * xk;
                        s = s + cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + s;
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    s = (s > cabs1(work[i]) / rwork[i])
                        ? s : cabs1(work[i]) / rwork[i];
                } else {
                    s = (s > (cabs1(work[i]) + safe1) / (rwork[i] + safe1))
                        ? s : (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
                }
            }
            berr[j] = s;

            /* Test stopping criterion. Continue iterating if
             *   1) The residual BERR(J) is larger than machine epsilon, and
             *   2) BERR(J) decreased by at least a factor of 2 during the
             *      last iteration, and
             *   3) At most ITMAX iterations tried. */
            if (!(berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX)) {
                break;
            }

            /* Update solution and try again. */
            zhetrs(uplo, n, 1, AF, ldaf, ipiv, work, n, &linfo);
            cblas_zaxpy(n, &ONE, work, 1, &X[j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
        }

        /* Bound error from formula
         *
         *   norm(X - XTRUE) / norm(X) .le. FERR =
         *   norm( abs(inv(A))*
         *      ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X)
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
         *
         * Use ZLACN2 to estimate the infinity-norm of the matrix
         *   inv(A) * diag(W),
         * where W = abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) */

        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        kase = 0;
        while (1) {
            zlacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                /* Multiply by diag(W)*inv(A**H). */
                zhetrs(uplo, n, 1, AF, ldaf, ipiv, work, n, &linfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
            } else {
                /* Multiply by inv(A)*diag(W). */
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
                zhetrs(uplo, n, 1, AF, ldaf, ipiv, work, n, &linfo);
            }
        }

        /* Normalize error. */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > cabs1(X[i + j * ldx])) ? lstres : cabs1(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
