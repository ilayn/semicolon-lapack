/**
 * @file dqrt16.c
 * @brief DQRT16 computes the residual for a solution of a system of linear
 *        equations A*x = b or A'*x = b.
 *
 * Faithful port of LAPACK TESTING/LIN/dqrt16.f
 *
 * RESID = norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS )
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* Forward declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);

/**
 * DQRT16 computes the residual for a solution of a system of linear
 * equations A*x = b or A'*x = b:
 *     RESID = norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS )
 *
 * @param[in] trans
 *     Specifies the form of the system of equations:
 *     = 'N': A *x = b
 *     = 'T': A'*x = b, where A' is the transpose of A
 *     = 'C': A'*x = b, where A' is the transpose of A
 *
 * @param[in] m
 *     The number of rows of the matrix A. M >= 0.
 *
 * @param[in] n
 *     The number of columns of the matrix A. N >= 0.
 *
 * @param[in] nrhs
 *     The number of columns of B, the matrix of right hand sides.
 *     NRHS >= 0.
 *
 * @param[in] A
 *     The original M x N matrix A.
 *
 * @param[in] lda
 *     The leading dimension of the array A. LDA >= max(1, M).
 *
 * @param[in] X
 *     The computed solution vectors for the system of linear equations.
 *
 * @param[in] ldx
 *     The leading dimension of the array X. If TRANS = 'N', LDX >= max(1, N);
 *     if TRANS = 'T' or 'C', LDX >= max(1, M).
 *
 * @param[in,out] B
 *     On entry, the right hand side vectors for the system of linear equations.
 *     On exit, B is overwritten with the difference B - A*X.
 *
 * @param[in] ldb
 *     The leading dimension of the array B. If TRANS = 'N', LDB >= max(1, M);
 *     if TRANS = 'T' or 'C', LDB >= max(1, N).
 *
 * @param[out] rwork
 *     Workspace array of dimension (M).
 *
 * @param[out] resid
 *     The maximum over the number of right hand sides of
 *     norm(B - A*X) / ( max(m, n) * norm(A) * norm(X) * EPS ).
 */
void dqrt16(const char* trans, const int m, const int n, const int nrhs,
            const f64* A, const int lda,
            const f64* X, const int ldx,
            f64* B, const int ldb,
            f64* rwork, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int j, n1, n2;
    f64 anorm, bnorm, eps, xnorm;
    int tpsd;

    /* Quick exit if M = 0 or N = 0 or NRHS = 0 */
    if (m <= 0 || n <= 0 || nrhs == 0) {
        *resid = ZERO;
        return;
    }

    tpsd = (trans[0] == 'T' || trans[0] == 't' ||
            trans[0] == 'C' || trans[0] == 'c');

    if (tpsd) {
        anorm = dlange("I", m, n, A, lda, rwork);
        n1 = n;
        n2 = m;
    } else {
        anorm = dlange("1", m, n, A, lda, rwork);
        n1 = m;
        n2 = n;
    }

    eps = dlamch("E");

    /* Compute B - A*X (or B - A'*X) and store in B */
    cblas_dgemm(CblasColMajor,
                tpsd ? CblasTrans : CblasNoTrans,
                CblasNoTrans,
                n1, nrhs, n2, -ONE, A, lda, X, ldx, ONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / ( max(m, n) * norm(A) * norm(X) * EPS ) */
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_dasum(n1, &B[j * ldb], 1);
        xnorm = cblas_dasum(n2, &X[j * ldx], 1);
        if (anorm == ZERO && bnorm == ZERO) {
            /* Both zero, no error */
        } else if (anorm <= ZERO || xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            int maxmn = (m > n) ? m : n;
            f64 temp = ((bnorm / anorm) / xnorm) / ((f64)maxmn * eps);
            if (temp > *resid) {
                *resid = temp;
            }
        }
    }
}
