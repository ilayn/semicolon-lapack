/**
 * @file zherfs.c
 * @brief ZHERFS improves the computed solution to a system of linear
 *        equations with a Hermitian indefinite matrix and provides
 *        error bounds and backward error estimates.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHERFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian indefinite, and
 * provides error bounds and backward error estimates for the solution.
 *
 * @param[in]     uplo  = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     A     The Hermitian matrix A. If uplo = 'U', the leading
 *                      N-by-N upper triangular part contains the upper
 *                      triangular part of A. If uplo = 'L', the leading
 *                      N-by-N lower triangular part contains the lower
 *                      triangular part of A.
 *                      Complex array, dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     AF    The factored form of the matrix A. AF contains
 *                      the block diagonal matrix D and the multipliers used
 *                      to obtain the factor U or L from the factorization
 *                      A = U*D*U**H or A = L*D*L**H as computed by ZHETRF.
 *                      Complex array, dimension (ldaf, n).
 * @param[in]     ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @param[in]     ipiv  Details of the interchanges and the block structure of D
 *                      as determined by ZHETRF. Integer array, dimension (n).
 * @param[in]     B     The right hand side matrix B.
 *                      Complex array, dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[in,out] X     On entry, the solution matrix X, as computed by ZHETRS.
 *                      On exit, the improved solution matrix X.
 *                      Complex array, dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1, n).
 * @param[out]    ferr  The estimated forward error bound for each solution
 *                      vector X(j). Real array, dimension (nrhs).
 * @param[out]    berr  The componentwise relative backward error of each
 *                      solution vector X(j). Real array, dimension (nrhs).
 * @param[out]    work  Complex workspace array, dimension (2*n).
 * @param[out]    rwork Real workspace array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zherfs(
    const char* uplo,
    const int n,
    const int nrhs,
    const double complex* const restrict A,
    const int lda,
    const double complex* const restrict AF,
    const int ldaf,
    const int* const restrict ipiv,
    const double complex* const restrict B,
    const int ldb,
    double complex* const restrict X,
    const int ldx,
    double* const restrict ferr,
    double* const restrict berr,
    double complex* const restrict work,
    double* const restrict rwork,
    int* info)
{
    const int ITMAX = 5;
    const double ZERO = 0.0;
    const double complex ONE = CMPLX(1.0, 0.0);
    const double TWO = 2.0;
    const double THREE = 3.0;

    int upper;
    int count, i, j, k, kase, nz;
    double eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int linfo;
    double complex neg_one = CMPLX(-1.0, 0.0);

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
