/**
 * @file ssyrfs.c
 * @brief SSYRFS improves the computed solution to a system of linear
 *        equations with a symmetric indefinite matrix and provides
 *        error bounds and backward error estimates.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSYRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric indefinite, and
 * provides error bounds and backward error estimates for the solution.
 *
 * @param[in]     uplo  = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     A     The symmetric matrix A. If uplo = 'U', the leading
 *                      N-by-N upper triangular part contains the upper
 *                      triangular part of A. If uplo = 'L', the leading
 *                      N-by-N lower triangular part contains the lower
 *                      triangular part of A.
 *                      Double precision array, dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     AF    The factored form of the matrix A. AF contains
 *                      the block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L from the factorization
 *                      A = U*D*U**T or A = L*D*L**T as computed by SSYTRF.
 *                      Double precision array, dimension (ldaf, n).
 * @param[in]     ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @param[in]     ipiv  Details of the interchanges and the block structure of D
 *                      as determined by SSYTRF. Integer array, dimension (n).
 * @param[in]     B     The right hand side matrix B.
 *                      Double precision array, dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] X     On entry, the solution matrix X, as computed by SSYTRS.
 *                      On exit, the improved solution matrix X.
 *                      Double precision array, dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1, n).
 * @param[out]    ferr  The estimated forward error bound for each solution
 *                      vector X(j). Double precision array, dimension (nrhs).
 * @param[out]    berr  The componentwise relative backward error of each
 *                      solution vector X(j). Double precision array, dimension (nrhs).
 * @param[out]    work  Double precision array, dimension (3*n).
 * @param[out]    iwork Integer array, dimension (n).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value
 */
void ssyrfs(
    const char* uplo,
    const int n,
    const int nrhs,
    const float* const restrict A,
    const int lda,
    const float* const restrict AF,
    const int ldaf,
    const int* const restrict ipiv,
    const float* const restrict B,
    const int ldb,
    float* const restrict X,
    const int ldx,
    float* const restrict ferr,
    float* const restrict berr,
    float* const restrict work,
    int* const restrict iwork,
    int* info)
{
    const int ITMAX = 5;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float THREE = 3.0f;

    int upper;
    int count, i, j, k, kase, nz;
    float eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int linfo;

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
        xerbla("SSYRFS", -(*info));
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
    eps = FLT_EPSILON;
    safmin = FLT_MIN;
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    /* Do for each right hand side. */
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        /* Iterative refinement loop. */
        while (1) {
            /* Compute residual R = B - A * X.
             * For symmetric A, DSYMV computes A*X directly. */
            cblas_scopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_ssymv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, -ONE, A, lda, &X[j * ldx], 1, ONE, &work[n], 1);

            /* Compute componentwise relative backward error from formula
             *   max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
             *
             * where abs(Z) is the componentwise absolute value of the matrix
             * or vector Z. */

            for (i = 0; i < n; i++) {
                work[i] = fabsf(B[i + j * ldb]);
            }

            /* Compute abs(A)*abs(X) + abs(B).
             * Uses symmetric structure: upper or lower triangle only. */
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabsf(X[k + j * ldx]);
                    for (i = 0; i < k; i++) {
                        work[i] += fabsf(A[i + k * lda]) * xk;
                        s += fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                    }
                    work[k] += fabsf(A[k + k * lda]) * xk + s;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabsf(X[k + j * ldx]);
                    work[k] += fabsf(A[k + k * lda]) * xk;
                    for (i = k + 1; i < n; i++) {
                        work[i] += fabsf(A[i + k * lda]) * xk;
                        s += fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                    }
                    work[k] += s;
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    float tmp = fabsf(work[n + i]) / work[i];
                    if (tmp > s) s = tmp;
                } else {
                    float tmp = (fabsf(work[n + i]) + safe1) / (work[i] + safe1);
                    if (tmp > s) s = tmp;
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
            ssytrs(uplo, n, 1, AF, ldaf, ipiv, &work[n], n, &linfo);
            cblas_saxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
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
         *   abs(Z) is the componentwise absolute value of the matrix or vector Z
         *   NZ is the maximum number of nonzeros in any row of A, plus 1
         *   EPS is machine epsilon
         *
         * The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B))
         * is incremented by SAFE1 if the i-th component of
         * abs(A)*abs(X) + abs(B) is less than SAFE2.
         *
         * Use SLACN2 to estimate the infinity-norm of the matrix
         *   inv(A) * diag(W),
         * where W = abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) */

        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        isave[0] = 0;
        isave[1] = 0;
        isave[2] = 0;
        while (1) {
            slacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                /* Multiply by diag(W)*inv(A**T).
                 * Since A is symmetric, inv(A**T) = inv(A). */
                ssytrs(uplo, n, 1, AF, ldaf, ipiv, &work[n], n, &linfo);
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else {
                /* Multiply by inv(A)*diag(W). */
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                ssytrs(uplo, n, 1, AF, ldaf, ipiv, &work[n], n, &linfo);
            }
        }

        /* Normalize error. */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            float tmp = fabsf(X[i + j * ldx]);
            if (tmp > lstres) lstres = tmp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
