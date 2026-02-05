/**
 * @file dqrt17.c
 * @brief DQRT17 computes the ratio norm(R'*op(A)) / (norm(A) * alpha * max(M,N,NRHS) * EPS).
 *
 * Faithful port of LAPACK TESTING/LIN/dqrt17.f
 *
 * This routine checks that the residual R = B - op(A)*X is orthogonal to
 * the column space of A (for least squares problems).
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* Forward declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlascl(const char* type, const int kl, const int ku,
                   const double cfrom, const double cto,
                   const int m, const int n, double* A, const int lda,
                   int* info);
extern void xerbla(const char* srname, const int info);

/**
 * DQRT17 computes the ratio
 *
 *    norm(R' * op(A)) / ( norm(A) * alpha * max(M,N,NRHS) * EPS ),
 *
 * where R = B - op(A)*X, op(A) is A or A', depending on TRANS, EPS
 * is the machine epsilon, and
 *
 *    alpha = norm(B) if IRESID = 1 (zero-residual problem)
 *    alpha = norm(R) if IRESID = 2 (otherwise).
 *
 * The norm used is the 1-norm.
 *
 * @param[in] trans
 *     Specifies whether or not the transpose of A is used.
 *     = 'N': No transpose, op(A) = A.
 *     = 'T': Transpose, op(A) = A'.
 *
 * @param[in] iresid
 *     IRESID = 1 indicates zero-residual problem.
 *     IRESID = 2 indicates non-zero residual.
 *
 * @param[in] m
 *     The number of rows of the matrix A.
 *     If TRANS = 'N', the number of rows of the matrix B.
 *     If TRANS = 'T', the number of rows of the matrix X.
 *
 * @param[in] n
 *     The number of columns of the matrix A.
 *     If TRANS = 'N', the number of rows of the matrix X.
 *     If TRANS = 'T', the number of rows of the matrix B.
 *
 * @param[in] nrhs
 *     The number of columns of the matrices X and B.
 *
 * @param[in] A
 *     The m-by-n matrix A.
 *
 * @param[in] lda
 *     The leading dimension of the array A. LDA >= M.
 *
 * @param[in] X
 *     If TRANS = 'N', the n-by-nrhs matrix X.
 *     If TRANS = 'T', the m-by-nrhs matrix X.
 *
 * @param[in] ldx
 *     The leading dimension of the array X.
 *     If TRANS = 'N', LDX >= N.
 *     If TRANS = 'T', LDX >= M.
 *
 * @param[in] B
 *     If TRANS = 'N', the m-by-nrhs matrix B.
 *     If TRANS = 'T', the n-by-nrhs matrix B.
 *
 * @param[in] ldb
 *     The leading dimension of the array B.
 *     If TRANS = 'N', LDB >= M.
 *     If TRANS = 'T', LDB >= N.
 *
 * @param[out] C
 *     Workspace array of dimension (LDB, NRHS).
 *
 * @param[out] work
 *     Workspace array of dimension (LWORK).
 *
 * @param[in] lwork
 *     The length of the array WORK. LWORK >= NRHS*(M+N).
 *
 * @return
 *     The computed ratio.
 */
double dqrt17(const char* trans, const int iresid,
              const int m, const int n, const int nrhs,
              const double* A, const int lda,
              const double* X, const int ldx,
              const double* B, const int ldb,
              double* C,
              double* work, const int lwork)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int info, iscl, ncols, nrows;
    double err, norma, normb, normrs, smlnum;
    double rwork[1];
    int tpsd;

    tpsd = (trans[0] == 'T' || trans[0] == 't');

    if (trans[0] == 'N' || trans[0] == 'n') {
        nrows = m;
        ncols = n;
    } else if (tpsd) {
        nrows = n;
        ncols = m;
    } else {
        xerbla("DQRT17", 1);
        return ZERO;
    }

    if (lwork < ncols * nrhs) {
        xerbla("DQRT17", 13);
        return ZERO;
    }

    if (m <= 0 || n <= 0 || nrhs <= 0) {
        return ZERO;
    }

    norma = dlange("O", m, n, A, lda, rwork);
    smlnum = dlamch("S") / dlamch("P");
    iscl = 0;

    /* Compute residual and scale it: C = B - op(A)*X */
    dlacpy("A", nrows, nrhs, B, ldb, C, ldb);
    cblas_dgemm(CblasColMajor,
                tpsd ? CblasTrans : CblasNoTrans,
                CblasNoTrans,
                nrows, nrhs, ncols, -ONE, A, lda, X, ldx, ONE, C, ldb);

    normrs = dlange("M", nrows, nrhs, C, ldb, rwork);
    if (normrs > smlnum) {
        iscl = 1;
        dlascl("G", 0, 0, normrs, ONE, nrows, nrhs, C, ldb, &info);
    }

    /* Compute R' * op(A) */
    cblas_dgemm(CblasColMajor, CblasTrans,
                tpsd ? CblasTrans : CblasNoTrans,
                nrhs, ncols, nrows, ONE, C, ldb, A, lda, ZERO, work, nrhs);

    /* Compute and properly scale error */
    err = dlange("O", nrhs, ncols, work, nrhs, rwork);
    if (norma != ZERO) {
        err = err / norma;
    }

    if (iscl == 1) {
        err = err * normrs;
    }

    if (iresid == 1) {
        normb = dlange("O", nrows, nrhs, B, ldb, rwork);
        if (normb != ZERO) {
            err = err / normb;
        }
    } else {
        if (normrs != ZERO) {
            err = err / normrs;
        }
    }

    int maxmnr = m;
    if (n > maxmnr) maxmnr = n;
    if (nrhs > maxmnr) maxmnr = nrhs;

    return err / (dlamch("E") * (double)maxmnr);
}
