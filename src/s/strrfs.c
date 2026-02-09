/** @file strrfs.c
 * @brief STRRFS provides error bounds for triangular solve. */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STRRFS provides error bounds and backward error estimates for the
 * solution to a system of linear equations with a triangular
 * coefficient matrix.
 *
 * The solution matrix X must be computed by STRTRS or some other
 * means before entering this routine.  STRRFS does not do iterative
 * refinement because doing so cannot improve the backward error.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  Specifies the form of the system of equations:
 *                       = 'N': A * X = B  (No transpose)
 *                       = 'T': A**T * X = B  (Transpose)
 *                       = 'C': A**H * X = B  (Conjugate transpose = Transpose)
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     A      The triangular matrix A. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[in]     X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1, n).
 * @param[out]    ferr   The estimated forward error bound for each solution
 *                       vector. Array of dimension (nrhs).
 * @param[out]    berr   The componentwise relative backward error.
 *                       Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -k, the k-th argument had an illegal value
 */
void strrfs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const float* const restrict A,
    const int lda,
    const float* const restrict B,
    const int ldb,
    const float* const restrict X,
    const int ldx,
    float* const restrict ferr,
    float* const restrict berr,
    float* const restrict work,
    int* const restrict iwork,
    int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    /* Test the input parameters */
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    int notran = (trans[0] == 'N' || trans[0] == 'n');
    int nounit = (diag[0] == 'N' || diag[0] == 'n');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't')
               && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("STRRFS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        for (int j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */
    int nz = n + 1;
    float eps = slamch("E");
    float safmin = slamch("S");
    float safe1 = nz * safmin;
    float safe2 = safe1 / eps;

    /* Set up CBLAS enums */
    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = notran ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transt = notran ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG cblas_siag = nounit ? CblasNonUnit : CblasUnit;

    /* Do for each right hand side */
    for (int j = 0; j < nrhs; j++) {

        /*
         * Compute residual R = B - op(A) * X,
         * where op(A) = A or A**T, depending on TRANS.
         *
         * Copy X(:,j) into work[n..2n-1], then compute
         * work[n..2n-1] = op(A) * X(:,j) via dtrmv,
         * then work[n..2n-1] = work[n..2n-1] - B(:,j)
         * so work[n..2n-1] = op(A)*X - B = -(B - op(A)*X) = -R
         * Actually the Fortran does: work(n+1) = X, dtrmv, daxpy(-1, B, work(n+1))
         * so work(n+1:2n) = op(A)*X - B
         * The sign does not matter for the absolute value usage below.
         */
        cblas_scopy(n, &X[j * ldx], 1, &work[n], 1);
        cblas_strmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_siag,
                    n, A, lda, &work[n], 1);
        cblas_saxpy(n, -ONE, &B[j * ldb], 1, &work[n], 1);

        /*
         * Compute componentwise relative backward error from formula:
         *
         * max(i) ( |R(i)| / ( |op(A)|*|X| + |B| )(i) )
         *
         * where |Z| is the componentwise absolute value of the matrix
         * or vector Z.
         */

        /* Initialize work[0..n-1] = |B(:,j)| */
        for (int i = 0; i < n; i++) {
            work[i] = fabsf(B[i + j * ldb]);
        }

        if (notran) {
            /* Compute |A|*|X| + |B| */
            if (upper) {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        float xk = fabsf(X[k + j * ldx]);
                        for (int i = 0; i <= k; i++) {
                            work[i] += fabsf(A[i + k * lda]) * xk;
                        }
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        float xk = fabsf(X[k + j * ldx]);
                        for (int i = 0; i < k; i++) {
                            work[i] += fabsf(A[i + k * lda]) * xk;
                        }
                        work[k] += xk;
                    }
                }
            } else {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        float xk = fabsf(X[k + j * ldx]);
                        for (int i = k; i < n; i++) {
                            work[i] += fabsf(A[i + k * lda]) * xk;
                        }
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        float xk = fabsf(X[k + j * ldx]);
                        for (int i = k + 1; i < n; i++) {
                            work[i] += fabsf(A[i + k * lda]) * xk;
                        }
                        work[k] += xk;
                    }
                }
            }
        } else {
            /* Compute |A**T|*|X| + |B| */
            if (upper) {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        float s = ZERO;
                        for (int i = 0; i <= k; i++) {
                            s += fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                        }
                        work[k] += s;
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        float s = fabsf(X[k + j * ldx]);
                        for (int i = 0; i < k; i++) {
                            s += fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                        }
                        work[k] += s;
                    }
                }
            } else {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        float s = ZERO;
                        for (int i = k; i < n; i++) {
                            s += fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                        }
                        work[k] += s;
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        float s = fabsf(X[k + j * ldx]);
                        for (int i = k + 1; i < n; i++) {
                            s += fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                        }
                        work[k] += s;
                    }
                }
            }
        }

        /* Compute BERR(j) */
        float s = ZERO;
        for (int i = 0; i < n; i++) {
            if (work[i] > safe2) {
                float tmp = fabsf(work[n + i]) / work[i];
                if (tmp > s) s = tmp;
            } else {
                float tmp = (fabsf(work[n + i]) + safe1) / (work[i] + safe1);
                if (tmp > s) s = tmp;
            }
        }
        berr[j] = s;

        /*
         * Bound error from formula:
         *
         * norm(X - XTRUE) / norm(X) <= FERR =
         * norm( |inv(op(A))| * ( |R| + NZ*EPS*( |op(A)|*|X|+|B| ))) / norm(X)
         *
         * where
         *   norm(Z) is the magnitude of the largest component of Z
         *   inv(op(A)) is the inverse of op(A)
         *   |Z| is the componentwise absolute value of the matrix or vector Z
         *   NZ is the maximum number of nonzeros in any row of A, plus 1
         *   EPS is machine epsilon
         *
         * The i-th component of |R|+NZ*EPS*(|op(A)|*|X|+|B|)
         * is incremented by SAFE1 if the i-th component of
         * |op(A)|*|X| + |B| is less than SAFE2.
         *
         * Use SLACN2 to estimate the infinity-norm of the matrix
         *   inv(op(A)) * diag(W),
         * where W = |R| + NZ*EPS*( |op(A)|*|X|+|B| )
         */
        for (int i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        int kase = 0;
        int isave[3] = {0, 0, 0};
        for (;;) {
            slacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) break;

            if (kase == 1) {
                /* Multiply by diag(W)*inv(op(A)**T) */
                cblas_strsv(CblasColMajor, cblas_uplo, cblas_transt,
                            cblas_siag, n, A, lda, &work[n], 1);
                for (int i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W) */
                for (int i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                cblas_strsv(CblasColMajor, cblas_uplo, cblas_trans,
                            cblas_siag, n, A, lda, &work[n], 1);
            }
        }

        /* Normalize error */
        float lstres = ZERO;
        for (int i = 0; i < n; i++) {
            float tmp = fabsf(X[i + j * ldx]);
            if (tmp > lstres) lstres = tmp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
