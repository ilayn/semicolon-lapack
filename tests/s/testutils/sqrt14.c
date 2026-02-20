/**
 * @file sqrt14.c
 * @brief SQRT14 checks whether X is in the row space of A or A'.
 *
 * Faithful port of LAPACK TESTING/LIN/sqrt14.f
 *
 * It does so by scaling both X and A such that their norms are in the range
 * [sqrt(eps), 1/sqrt(eps)], then computing a QR factorization of [A,X]
 * (if TRANS = 'T') or an LQ factorization of [A',X]' (if TRANS = 'N'),
 * and returning the norm of the trailing triangle, scaled by
 * MAX(M,N,NRHS)*eps.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* Forward declarations */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slascl(const char* type, const int kl, const int ku,
                   const f32 cfrom, const f32 cto,
                   const int m, const int n, f32* A, const int lda,
                   int* info);
extern void sgeqr2(const int m, const int n, f32* A, const int lda,
                   f32* tau, f32* work, int* info);
extern void sgelq2(const int m, const int n, f32* A, const int lda,
                   f32* tau, f32* work, int* info);
extern void xerbla(const char* srname, const int info);

/**
 * SQRT14 checks whether X is in the row space of A or A'.
 *
 * @param[in] trans
 *     = 'N': No transpose, check for X in the row space of A
 *     = 'T': Transpose, check for X in the row space of A'.
 *
 * @param[in] m
 *     The number of rows of the matrix A.
 *
 * @param[in] n
 *     The number of columns of the matrix A.
 *
 * @param[in] nrhs
 *     The number of right hand sides, i.e., the number of columns of X.
 *
 * @param[in] A
 *     The M-by-N matrix A.
 *
 * @param[in] lda
 *     The leading dimension of the array A.
 *
 * @param[in] X
 *     If TRANS = 'N', the N-by-NRHS matrix X.
 *     If TRANS = 'T', the M-by-NRHS matrix X.
 *
 * @param[in] ldx
 *     The leading dimension of the array X.
 *
 * @param[out] work
 *     Workspace array.
 *     If TRANS = 'N', LWORK >= (M+NRHS)*(N+2);
 *     if TRANS = 'T', LWORK >= (N+NRHS)*(M+2).
 *
 * @param[in] lwork
 *     Length of workspace array.
 *
 * @return
 *     The computed residual: norm of trailing triangle / (max(M,N,NRHS) * eps)
 */
f32 sqrt14(const char* trans, const int m, const int n, const int nrhs,
              const f32* A, const int lda, const f32* X, const int ldx,
              f32* work, const int lwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int tpsd;
    int i, info, j, ldwork;
    f32 anrm, err, xnrm;
    f32 rwork[1];

    if (trans[0] == 'N' || trans[0] == 'n') {
        ldwork = m + nrhs;
        tpsd = 0;
        if (lwork < (m + nrhs) * (n + 2)) {
            xerbla("SQRT14", 10);
            return ZERO;
        } else if (n <= 0 || nrhs <= 0) {
            return ZERO;
        }
    } else if (trans[0] == 'T' || trans[0] == 't') {
        ldwork = m;
        tpsd = 1;
        if (lwork < (n + nrhs) * (m + 2)) {
            xerbla("SQRT14", 10);
            return ZERO;
        } else if (m <= 0 || nrhs <= 0) {
            return ZERO;
        }
    } else {
        xerbla("SQRT14", 1);
        return ZERO;
    }

    /* Copy and scale A */
    slacpy("A", m, n, A, lda, work, ldwork);
    anrm = slange("M", m, n, work, ldwork, rwork);
    if (anrm != ZERO) {
        slascl("G", 0, 0, anrm, ONE, m, n, work, ldwork, &info);
    }

    /* Copy X or X' into the right place and scale it */
    if (tpsd) {
        /* Copy X into columns n:n+nrhs-1 of work (0-based: column n) */
        slacpy("A", m, nrhs, X, ldx, &work[n * ldwork], ldwork);
        xnrm = slange("M", m, nrhs, &work[n * ldwork], ldwork, rwork);
        if (xnrm != ZERO) {
            slascl("G", 0, 0, xnrm, ONE, m, nrhs, &work[n * ldwork],
                   ldwork, &info);
        }

        /* Compute QR factorization of [A, X] */
        int minmn = (m < n + nrhs) ? m : n + nrhs;
        sgeqr2(m, n + nrhs, work, ldwork,
               &work[ldwork * (n + nrhs)],
               &work[ldwork * (n + nrhs) + minmn],
               &info);

        /* Compute largest entry in upper triangle of work(n:m-1, n:n+nrhs-1)
           (0-based: rows n to min(m,j+1)-1 for column j from n to n+nrhs-1) */
        err = ZERO;
        for (j = n; j < n + nrhs; j++) {
            int iend = (m < j + 1) ? m : j + 1;  /* min(m, j+1) */
            for (i = n; i < iend; i++) {
                f32 val = fabsf(work[i + j * ldwork]);
                if (val > err) err = val;
            }
        }
    } else {
        /* Copy X' into rows m:m+nrhs-1 of work */
        for (i = 0; i < n; i++) {
            for (j = 0; j < nrhs; j++) {
                work[m + j + i * ldwork] = X[i + j * ldx];
            }
        }

        xnrm = slange("M", nrhs, n, &work[m], ldwork, rwork);
        if (xnrm != ZERO) {
            slascl("G", 0, 0, xnrm, ONE, nrhs, n, &work[m], ldwork, &info);
        }

        /* Compute LQ factorization of work */
        sgelq2(ldwork, n, work, ldwork,
               &work[ldwork * n],
               &work[ldwork * (n + 1)],
               &info);

        /* Compute largest entry in lower triangle of work(m:m+nrhs-1, m:n-1) */
        err = ZERO;
        for (j = m; j < n; j++) {
            for (i = j; i < ldwork; i++) {
                f32 val = fabsf(work[i + j * ldwork]);
                if (val > err) err = val;
            }
        }
    }

    /* Compute the result */
    int maxmnr = m;
    if (n > maxmnr) maxmnr = n;
    if (nrhs > maxmnr) maxmnr = nrhs;

    return err / ((f32)maxmnr * slamch("E"));
}
