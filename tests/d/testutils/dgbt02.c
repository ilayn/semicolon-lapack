/**
 * @file dgbt02.c
 * @brief DGBT02 computes the residual for a banded system solution.
 *
 * Port of LAPACK TESTING/LIN/dgbt02.f
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

/**
 * DGBT02 computes the residual for a solution of a banded system of
 * equations op(A)*X = B:
 *    RESID = norm(B - op(A)*X) / ( norm(op(A)) * norm(X) * EPS ),
 * where op(A) = A or A**T, depending on TRANS, and EPS is the
 * machine epsilon.
 * The norm used is the 1-norm.
 *
 * @param[in] trans  'N': A * X = B (No transpose)
 *                   'T': A**T * X = B (Transpose)
 *                   'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in] m      The number of rows of A. m >= 0.
 * @param[in] n      The number of columns of A. n >= 0.
 * @param[in] kl     The number of subdiagonals within the band. kl >= 0.
 * @param[in] ku     The number of superdiagonals within the band. ku >= 0.
 * @param[in] nrhs   The number of columns of B. nrhs >= 0.
 * @param[in] A      The original band matrix in band storage. Dimension (lda, n).
 * @param[in] lda    The leading dimension of A. lda >= max(1, kl+ku+1).
 * @param[in] X      The computed solution vectors. Dimension (ldx, nrhs).
 * @param[in] ldx    The leading dimension of X. If trans='N', ldx >= max(1,n);
 *                   otherwise ldx >= max(1,m).
 * @param[in,out] B  On entry, the right hand side vectors.
 *                   On exit, overwritten with B - op(A)*X.
 *                   Dimension (ldb, nrhs).
 * @param[in] ldb    The leading dimension of B. If trans='N', ldb >= max(1,m);
 *                   otherwise ldb >= max(1,n).
 * @param[out] rwork Workspace, dimension (m) when trans='T' or 'C'.
 * @param[out] resid The maximum over NRHS of
 *                   norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS).
 */
void dgbt02(const char* trans, int m, int n, int kl, int ku, int nrhs,
            const f64* A, int lda,
            const f64* X, int ldx,
            f64* B, int ldb,
            f64* rwork,
            f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Quick return if m = 0 or n = 0 or nrhs = 0 */
    if (m <= 0 || n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    /* Determine EPS and the norm of A. */
    f64 eps = dlamch("Epsilon");
    f64 anorm = ZERO;
    int notran = (trans[0] == 'N' || trans[0] == 'n');

    if (notran) {
        /* Find norm1(A). */
        int kd = ku;  /* Row index offset for diagonal */
        for (int j = 0; j < n; j++) {
            int i1 = (kd + 1 - j - 1 > 0) ? kd + 1 - j - 1 : 0;
            int i2_excl = (kd + m - j < kl + kd + 1) ? kd + m - j : kl + kd + 1;
            if (i2_excl > i1) {
                f64 temp = cblas_dasum(i2_excl - i1, &A[i1 + j * lda], 1);
                if (temp > anorm || isnan(temp)) {
                    anorm = temp;
                }
            }
        }
    } else {
        /* Find normI(A). */
        for (int i = 0; i < m; i++) {
            rwork[i] = ZERO;
        }
        for (int j = 0; j < n; j++) {
            int kd = ku - j;
            for (int i = (j - ku > 0 ? j - ku : 0); i < (j + kl + 1 < m ? j + kl + 1 : m); i++) {
                rwork[i] += fabs(A[(kd + i) + j * lda]);
            }
        }
        for (int i = 0; i < m; i++) {
            f64 temp = rwork[i];
            if (temp > anorm || isnan(temp)) {
                anorm = temp;
            }
        }
    }

    /* Exit with resid = 1/EPS if anorm = 0. */
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    int n1 = notran ? m : n;

    /* Compute B - op(A)*X */
    for (int j = 0; j < nrhs; j++) {
        cblas_dgbmv(CblasColMajor,
                   notran ? CblasNoTrans : CblasTrans,
                   m, n, kl, ku,
                   -ONE, A, lda,
                   &X[j * ldx], 1,
                   ONE, &B[j * ldb], 1);
    }

    /* Compute the maximum over the number of right hand sides of
       norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS). */
    *resid = ZERO;
    for (int j = 0; j < nrhs; j++) {
        f64 bnorm = cblas_dasum(n1, &B[j * ldb], 1);
        f64 xnorm = cblas_dasum(n1, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f64 temp = ((bnorm / anorm) / xnorm) / eps;
            if (temp > *resid) {
                *resid = temp;
            }
        }
    }
}
