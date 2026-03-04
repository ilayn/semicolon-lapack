/**
 * @file zqrt17.c
 * @brief ZQRT17 computes the ratio norm(R^H*op(A)) / (norm(A) * alpha * max(M,N,NRHS) * EPS).
 *
 * Faithful port of LAPACK TESTING/LIN/zqrt17.f
 *
 * This routine checks that the residual R = B - op(A)*X is orthogonal to
 * the column space of A (for least squares problems).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZQRT17 computes the ratio
 *
 *    norm(R^H * op(A)) / ( norm(A) * alpha * max(M,N,NRHS) * EPS ),
 *
 * where R = B - op(A)*X, op(A) is A or A^H, depending on TRANS, EPS
 * is the machine epsilon, and
 *
 *    alpha = norm(B) if IRESID = 1 (zero-residual problem)
 *    alpha = norm(R) if IRESID = 2 (otherwise).
 *
 * The norm used is the 1-norm.
 *
 * @param[in] trans
 *     Specifies whether or not the conjugate transpose of A is used.
 *     = 'N': No transpose, op(A) = A.
 *     = 'C': Conjugate transpose, op(A) = A^H.
 *
 * @param[in] iresid
 *     IRESID = 1 indicates zero-residual problem.
 *     IRESID = 2 indicates non-zero residual.
 *
 * @param[in] m
 *     The number of rows of the matrix A.
 *     If TRANS = 'N', the number of rows of the matrix B.
 *     If TRANS = 'C', the number of rows of the matrix X.
 *
 * @param[in] n
 *     The number of columns of the matrix A.
 *     If TRANS = 'N', the number of rows of the matrix X.
 *     If TRANS = 'C', the number of rows of the matrix B.
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
 *     If TRANS = 'C', the m-by-nrhs matrix X.
 *
 * @param[in] ldx
 *     The leading dimension of the array X.
 *     If TRANS = 'N', LDX >= N.
 *     If TRANS = 'C', LDX >= M.
 *
 * @param[in] B
 *     If TRANS = 'N', the m-by-nrhs matrix B.
 *     If TRANS = 'C', the n-by-nrhs matrix B.
 *
 * @param[in] ldb
 *     The leading dimension of the array B.
 *     If TRANS = 'N', LDB >= M.
 *     If TRANS = 'C', LDB >= N.
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
f64 zqrt17(const char* trans, const INT iresid,
              const INT m, const INT n, const INT nrhs,
              const c128* A, const INT lda,
              const c128* X, const INT ldx,
              const c128* B, const INT ldb,
              c128* C,
              c128* work, const INT lwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT info, iscl, ncols, nrows;
    f64 err, norma, normb, normrs, smlnum;
    f64 rwork[1];
    INT tpsd;

    tpsd = (trans[0] == 'C' || trans[0] == 'c');

    if (trans[0] == 'N' || trans[0] == 'n') {
        nrows = m;
        ncols = n;
    } else if (tpsd) {
        nrows = n;
        ncols = m;
    } else {
        xerbla("ZQRT17", 1);
        return ZERO;
    }

    if (lwork < ncols * nrhs) {
        xerbla("ZQRT17", 13);
        return ZERO;
    }

    if (m <= 0 || n <= 0 || nrhs <= 0) {
        return ZERO;
    }

    norma = zlange("O", m, n, A, lda, rwork);
    smlnum = dlamch("S") / dlamch("P");
    iscl = 0;

    /* compute residual and scale it */
    zlacpy("A", nrows, nrhs, B, ldb, C, ldb);
    cblas_zgemm(CblasColMajor,
                tpsd ? CblasConjTrans : CblasNoTrans,
                CblasNoTrans,
                nrows, nrhs, ncols, &CNEGONE, A, lda, X, ldx, &CONE, C, ldb);

    normrs = zlange("M", nrows, nrhs, C, ldb, rwork);
    if (normrs > smlnum) {
        iscl = 1;
        zlascl("G", 0, 0, normrs, ONE, nrows, nrhs, C, ldb, &info);
    }

    /* compute R^H * op(A) */
    cblas_zgemm(CblasColMajor, CblasConjTrans,
                tpsd ? CblasConjTrans : CblasNoTrans,
                nrhs, ncols, nrows, &CONE, C, ldb, A, lda, &CZERO, work, nrhs);

    /* compute and properly scale error */
    err = zlange("O", nrhs, ncols, work, nrhs, rwork);
    if (norma != ZERO) {
        err = err / norma;
    }

    if (iscl == 1) {
        err = err * normrs;
    }

    if (iresid == 1) {
        normb = zlange("O", nrows, nrhs, B, ldb, rwork);
        if (normb != ZERO) {
            err = err / normb;
        }
    } else {
        if (normrs != ZERO) {
            err = err / normrs;
        }
    }

    INT maxmnr = m;
    if (n > maxmnr) maxmnr = n;
    if (nrhs > maxmnr) maxmnr = nrhs;

    return err / (dlamch("E") * (f64)maxmnr);
}
