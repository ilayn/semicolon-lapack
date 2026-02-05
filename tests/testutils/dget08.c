/**
 * @file dget08.c
 * @brief DGET08 computes the residual for a solution of a system of linear equations.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "verify.h"

/**
 * DGET08 computes the residual for a solution of a system of linear
 * equations  A*x = b  or  A'*x = b:
 *    RESID = norm(B - A*X,inf) / ( norm(A,inf) * norm(X,inf) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     trans   Specifies the form of the system of equations:
 *                        = "N":  A *x = b
 *                        = "T":  A'*x = b, where A' is the transpose of A
 *                        = "C":  A'*x = b, where A' is the transpose of A
 * @param[in]     m       The number of rows of the matrix A.  m >= 0.
 * @param[in]     n       The number of columns of the matrix A.  n >= 0.
 * @param[in]     nrhs    The number of columns of B, the matrix of right hand sides.
 *                        nrhs >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original m x n matrix A.
 * @param[in]     lda     The leading dimension of the array A.  lda >= max(1,m).
 * @param[in]     X       Double precision array, dimension (ldx, nrhs).
 *                        The computed solution vectors for the system.
 * @param[in]     ldx     The leading dimension of the array X.  If trans = 'N',
 *                        ldx >= max(1,n); if trans = 'T' or 'C', ldx >= max(1,m).
 * @param[in,out] B       Double precision array, dimension (ldb, nrhs).
 *                        On entry, the right hand side vectors.
 *                        On exit, B is overwritten with the difference B - A*X.
 * @param[in]     ldb     The leading dimension of the array B.  If trans = 'N',
 *                        ldb >= max(1,m); if trans = 'T' or 'C', ldb >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (m).
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void dget08(const char* trans, const int m, const int n, const int nrhs,
            const double* A, const int lda, const double* X, const int ldx,
            double* B, const int ldb, double* rwork, double* resid)
{
    (void)rwork;
    int n1, n2;
    double anorm, bnorm, xnorm, eps;

    /* Quick exit if m = 0 or n = 0 or nrhs = 0 */
    if (m <= 0 || n <= 0 || nrhs == 0) {
        *resid = 0.0;
        return;
    }

    /* Determine dimensions based on transpose */
    if (trans[0] == 'T' || trans[0] == 't' || trans[0] == 'C' || trans[0] == 'c') {
        n1 = n;
        n2 = m;
    } else {
        n1 = m;
        n2 = n;
    }

    /* Get machine epsilon */
    eps = DBL_EPSILON;

    /* Compute infinity norm of A */
    /* ||A||_inf = max over rows of sum of absolute values */
    anorm = 0.0;
    for (int i = 0; i < n1; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n2; j++) {
            row_sum += fabs(A[i + j * lda]);
        }
        if (row_sum > anorm) anorm = row_sum;
    }

    /* Exit with resid = 1/eps if anorm = 0 */
    if (anorm <= 0.0) {
        *resid = 1.0 / eps;
        return;
    }

    /* Compute B - A*X (or B - A'*X) and store in B */
    /* B := -A*X + B  or  B := -A'*X + B */
    CBLAS_TRANSPOSE trans_cblas = CblasNoTrans;
    if (trans[0] == 'T' || trans[0] == 't') {
        trans_cblas = CblasTrans;
    } else if (trans[0] == 'C' || trans[0] == 'c') {
        trans_cblas = CblasConjTrans;
    }

    cblas_dgemm(CblasColMajor, trans_cblas, CblasNoTrans,
                n1, nrhs, n2, -1.0, A, lda, X, ldx, 1.0, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / (norm(A) * norm(X) * eps) */
    *resid = 0.0;
    for (int j = 0; j < nrhs; j++) {
        /* Compute infinity norm of column j of B */
        int idx = cblas_idamax(n1, &B[j * ldb], 1);
        bnorm = fabs(B[idx + j * ldb]);

        /* Compute infinity norm of column j of X */
        idx = cblas_idamax(n2, &X[j * ldx], 1);
        xnorm = fabs(X[idx + j * ldx]);

        if (xnorm <= 0.0) {
            *resid = 1.0 / eps;
        } else {
            double ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) *resid = ratio;
        }
    }
}
