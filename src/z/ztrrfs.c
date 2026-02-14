/** @file ztrrfs.c
 * @brief ZTRRFS provides error bounds for triangular solve. */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTRRFS provides error bounds and backward error estimates for the
 * solution to a system of linear equations with a triangular
 * coefficient matrix.
 *
 * The solution matrix X must be computed by ZTRTRS or some other
 * means before entering this routine.  ZTRRFS does not do iterative
 * refinement because doing so cannot improve the backward error.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  Specifies the form of the system of equations:
 *                       = 'N': A * X = B  (No transpose)
 *                       = 'T': A**T * X = B  (Transpose)
 *                       = 'C': A**H * X = B  (Conjugate transpose)
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
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void ztrrfs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const c128* restrict A,
    const int lda,
    const c128* restrict B,
    const int ldb,
    const c128* restrict X,
    const int ldx,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    f64* restrict rwork,
    int* info)
{
    const f64 ZERO = 0.0;

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
        xerbla("ZTRRFS", -(*info));
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
    f64 eps = dlamch("E");
    f64 safmin = dlamch("S");
    f64 safe1 = nz * safmin;
    f64 safe2 = safe1 / eps;

    /* Set up CBLAS enums */
    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'N' || trans[0] == 'n') {
        cblas_trans = CblasNoTrans;
    } else if (trans[0] == 'T' || trans[0] == 't') {
        cblas_trans = CblasTrans;
    } else {
        cblas_trans = CblasConjTrans;
    }
    CBLAS_TRANSPOSE cblas_transn = notran ? CblasNoTrans : CblasConjTrans;
    CBLAS_TRANSPOSE cblas_transt = notran ? CblasConjTrans : CblasNoTrans;
    CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;

    /* Do for each right hand side */
    for (int j = 0; j < nrhs; j++) {

        /*
         * Compute residual R = B - op(A) * X,
         * where op(A) = A, A**T, or A**H, depending on TRANS.
         */
        cblas_zcopy(n, &X[j * ldx], 1, work, 1);
        cblas_ztrmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, A, lda, work, 1);
        c128 neg_one = CMPLX(-1.0, 0.0);
        cblas_zaxpy(n, &neg_one, &B[j * ldb], 1, work, 1);

        /*
         * Compute componentwise relative backward error from formula
         *
         * max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
         *
         * where abs(Z) is the componentwise absolute value of the matrix
         * or vector Z.  If the i-th component of the denominator is less
         * than SAFE2, then SAFE1 is added to the i-th components of the
         * numerator and denominator before dividing.
         */

        for (int i = 0; i < n; i++) {
            rwork[i] = cabs1(B[i + j * ldb]);
        }

        if (notran) {

            /* Compute abs(A)*abs(X) + abs(B). */
            if (upper) {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        f64 xk = cabs1(X[k + j * ldx]);
                        for (int i = 0; i <= k; i++) {
                            rwork[i] += cabs1(A[i + k * lda]) * xk;
                        }
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        f64 xk = cabs1(X[k + j * ldx]);
                        for (int i = 0; i < k; i++) {
                            rwork[i] += cabs1(A[i + k * lda]) * xk;
                        }
                        rwork[k] += xk;
                    }
                }
            } else {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        f64 xk = cabs1(X[k + j * ldx]);
                        for (int i = k; i < n; i++) {
                            rwork[i] += cabs1(A[i + k * lda]) * xk;
                        }
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        f64 xk = cabs1(X[k + j * ldx]);
                        for (int i = k + 1; i < n; i++) {
                            rwork[i] += cabs1(A[i + k * lda]) * xk;
                        }
                        rwork[k] += xk;
                    }
                }
            }
        } else {

            /* Compute abs(A**H)*abs(X) + abs(B). */
            if (upper) {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        f64 s = ZERO;
                        for (int i = 0; i <= k; i++) {
                            s += cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] += s;
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        f64 s = cabs1(X[k + j * ldx]);
                        for (int i = 0; i < k; i++) {
                            s += cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] += s;
                    }
                }
            } else {
                if (nounit) {
                    for (int k = 0; k < n; k++) {
                        f64 s = ZERO;
                        for (int i = k; i < n; i++) {
                            s += cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] += s;
                    }
                } else {
                    for (int k = 0; k < n; k++) {
                        f64 s = cabs1(X[k + j * ldx]);
                        for (int i = k + 1; i < n; i++) {
                            s += cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] += s;
                    }
                }
            }
        }

        f64 s = ZERO;
        for (int i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                f64 tmp = cabs1(work[i]) / rwork[i];
                if (tmp > s) s = tmp;
            } else {
                f64 tmp = (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
                if (tmp > s) s = tmp;
            }
        }
        berr[j] = s;

        /*
         * Bound error from formula
         *
         * norm(X - XTRUE) / norm(X) .le. FERR =
         * norm( abs(inv(op(A)))*
         *    ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
         *
         * where
         *   norm(Z) is the magnitude of the largest component of Z
         *   inv(op(A)) is the inverse of op(A)
         *   abs(Z) is the componentwise absolute value of the matrix or
         *      vector Z
         *   NZ is the maximum number of nonzeros in any row of A, plus 1
         *   EPS is machine epsilon
         *
         * The i-th component of abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B))
         * is incremented by SAFE1 if the i-th component of
         * abs(op(A))*abs(X) + abs(B) is less than SAFE2.
         *
         * Use ZLACN2 to estimate the infinity-norm of the matrix
         *    inv(op(A)) * diag(W),
         * where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) )))
         */
        for (int i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        int kase = 0;
        int isave[3] = {0, 0, 0};
        for (;;) {
            zlacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase == 0) break;

            if (kase == 1) {
                /* Multiply by diag(W)*inv(op(A)**H). */
                cblas_ztrsv(CblasColMajor, cblas_uplo, cblas_transt,
                            cblas_diag, n, A, lda, work, 1);
                for (int i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W). */
                for (int i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
                cblas_ztrsv(CblasColMajor, cblas_uplo, cblas_transn,
                            cblas_diag, n, A, lda, work, 1);
            }
        }

        /* Normalize error. */
        f64 lstres = ZERO;
        for (int i = 0; i < n; i++) {
            f64 tmp = cabs1(X[i + j * ldx]);
            if (tmp > lstres) lstres = tmp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
