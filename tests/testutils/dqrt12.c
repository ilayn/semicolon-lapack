/**
 * @file dqrt12.c
 * @brief DQRT12 computes || svd(R) - s || / (||s|| * eps * max(M,N)).
 *
 * Port of LAPACK TESTING/LIN/dqrt12.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern void dlascl(const char* type, const int kl, const int ku,
                   const double cfrom, const double cto,
                   const int m, const int n, double* A, const int lda,
                   int* info);
extern void dgebd2(const int m, const int n, double* A, const int lda,
                   double* D, double* E, double* tauq, double* taup,
                   double* work, int* info);
extern void dbdsqr(const char* uplo, const int n, const int ncvt,
                   const int nru, const int ncc, double* D, double* E,
                   double* VT, const int ldvt, double* U, const int ldu,
                   double* C, const int ldc, double* work, int* info);

/**
 * DQRT12 computes the singular values of the upper trapezoid
 * of A(1:M,1:N) and returns the ratio
 *
 *    || svlues - s || / (||s|| * eps * max(M,N))
 *
 * @param[in]  m     The number of rows of the matrix A.
 * @param[in]  n     The number of columns of the matrix A.
 * @param[in]  A     Array (lda, n). The M-by-N matrix A. Only the upper
 *                   trapezoid is referenced.
 * @param[in]  lda   The leading dimension of the array A.
 * @param[in]  S     Array (min(m,n)). The singular values of the matrix A.
 * @param[out] work  Array (lwork). Workspace.
 * @param[in]  lwork The length of the array work.
 *
 * @return The test ratio || svd(R) - s || / (||s|| * eps * max(M,N)).
 */
double dqrt12(const int m, const int n, const double* A, const int lda,
              const double* S, double* work, const int lwork)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    int i, j, mn, info, iscl;
    double anrm, bignum, smlnum, nrmsvl;
    double dummy[1];

    /* Quick return if possible */
    mn = (m < n) ? m : n;
    if (mn <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    int lwork_min1 = m * n + 4 * mn + ((m > n) ? m : n);
    int lwork_min2 = m * n + 2 * mn + 4 * n;
    int lwork_min = (lwork_min1 > lwork_min2) ? lwork_min1 : lwork_min2;
    if (lwork < lwork_min) {
        return ZERO;
    }

    /* Compute ||S|| */
    nrmsvl = cblas_dnrm2(mn, S, 1);

    /* Copy upper triangle of A into work */
    dlaset("F", m, n, ZERO, ZERO, work, m);
    for (j = 0; j < n; j++) {
        int imax = (j + 1 < m) ? (j + 1) : m;
        for (i = 0; i < imax; i++) {
            work[j * m + i] = A[j * lda + i];
        }
    }

    /* Get machine parameters */
    smlnum = dlamch("S") / dlamch("P");
    bignum = ONE / smlnum;

    /* Scale work if max entry outside range [SMLNUM, BIGNUM] */
    anrm = dlange("M", m, n, work, m, dummy);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        dlascl("G", 0, 0, anrm, smlnum, m, n, work, m, &info);
        iscl = 1;
    } else if (anrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        dlascl("G", 0, 0, anrm, bignum, m, n, work, m, &info);
        iscl = 1;
    }

    if (anrm != ZERO) {
        /* Compute SVD of work */
        /* work(m*n:)     = D (diagonal)
         * work(m*n+mn:)  = E (superdiagonal)
         * work(m*n+2*mn:) = tauq
         * work(m*n+3*mn:) = taup
         * work(m*n+4*mn:) = workspace for dgebd2 */
        dgebd2(m, n, work, m,
               &work[m * n], &work[m * n + mn],
               &work[m * n + 2 * mn], &work[m * n + 3 * mn],
               &work[m * n + 4 * mn], &info);

        /* Compute singular values from bidiagonal form */
        dbdsqr("U", mn, 0, 0, 0,
               &work[m * n], &work[m * n + mn],
               dummy, mn, dummy, 1, dummy, mn,
               &work[m * n + 2 * mn], &info);

        if (iscl == 1) {
            if (anrm > bignum) {
                dlascl("G", 0, 0, bignum, anrm, mn, 1, &work[m * n], mn, &info);
            }
            if (anrm < smlnum) {
                dlascl("G", 0, 0, smlnum, anrm, mn, 1, &work[m * n], mn, &info);
            }
        }
    } else {
        for (i = 0; i < mn; i++) {
            work[m * n + i] = ZERO;
        }
    }

    /* Compare s and singular values of work: work(m*n:m*n+mn) -= S */
    cblas_daxpy(mn, -ONE, S, 1, &work[m * n], 1);

    /* Return || diff || / (eps * max(M,N)) / ||S|| */
    double result = cblas_dasum(mn, &work[m * n], 1) /
                    (dlamch("E") * (double)((m > n) ? m : n));
    if (nrmsvl != ZERO) {
        result /= nrmsvl;
    }

    return result;
}
