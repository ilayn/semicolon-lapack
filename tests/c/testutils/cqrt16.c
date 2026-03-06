/**
 * @file cqrt16.c
 * @brief CQRT16 computes the residual for a solution of a system of linear
 *        equations A*x = b or A'*x = b or A^H*x = b.
 *
 * Faithful port of LAPACK TESTING/LIN/cqrt16.f
 *
 * RESID = norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS )
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CQRT16 computes the residual for a solution of a system of linear
 * equations A*x = b or A'*x = b or A^H*x = b:
 *     RESID = norm(B - A*X) / ( max(m,n) * norm(A) * norm(X) * EPS )
 *
 * @param[in] trans
 *     Specifies the form of the system of equations:
 *     = 'N': A *x = b
 *     = 'T': A^T*x = b, where A^T is the transpose of A
 *     = 'C': A^H*x = b, where A^H is the conjugate transpose of A
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
 *     The leading dimension of the array B. IF TRANS = 'N',
 *     LDB >= max(1, M); if TRANS = 'T' or 'C', LDB >= max(1, N).
 *
 * @param[out] rwork
 *     Workspace array of dimension (M).
 *
 * @param[out] resid
 *     The maximum over the number of right hand sides of
 *     norm(B - A*X) / ( max(m, n) * norm(A) * norm(X) * EPS ).
 */
void cqrt16(const char* trans, const INT m, const INT n, const INT nrhs,
            const c64* A, const INT lda,
            const c64* X, const INT ldx,
            c64* B, const INT ldb,
            f32* rwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT j, n1, n2;
    f32 anorm, bnorm, eps, xnorm;

    /* Quick exit if M = 0 or N = 0 or NRHS = 0 */
    if (m <= 0 || n <= 0 || nrhs == 0) {
        *resid = ZERO;
        return;
    }

    if (trans[0] == 'T' || trans[0] == 't' ||
        trans[0] == 'C' || trans[0] == 'c') {
        anorm = clange("I", m, n, A, lda, rwork);
        n1 = n;
        n2 = m;
    } else {
        anorm = clange("1", m, n, A, lda, rwork);
        n1 = m;
        n2 = n;
    }

    eps = slamch("E");

    /* Compute B - A*X (or B - A'*X or B - A^H*X) and store in B */
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'T' || trans[0] == 't') {
        cblas_trans = CblasTrans;
    } else if (trans[0] == 'C' || trans[0] == 'c') {
        cblas_trans = CblasConjTrans;
    } else {
        cblas_trans = CblasNoTrans;
    }

    cblas_cgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                n1, nrhs, n2, &CNEGONE, A, lda, X, ldx, &CONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / ( max(m, n) * norm(A) * norm(X) * EPS ) */
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_scasum(n1, &B[j * ldb], 1);
        xnorm = cblas_scasum(n2, &X[j * ldx], 1);
        if (anorm == ZERO && bnorm == ZERO) {
            /* Both zero, no error */
        } else if (anorm <= ZERO || xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            INT maxmn = (m > n) ? m : n;
            f32 temp = ((bnorm / anorm) / xnorm) / ((f32)maxmn * eps);
            if (temp > *resid) {
                *resid = temp;
            }
        }
    }
}
