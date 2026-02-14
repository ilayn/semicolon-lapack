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
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern void dlascl(const char* type, const int kl, const int ku,
                   const f64 cfrom, const f64 cto,
                   const int m, const int n, f64* A, const int lda,
                   int* info);
extern void dgebd2(const int m, const int n, f64* A, const int lda,
                   f64* D, f64* E, f64* tauq, f64* taup,
                   f64* work, int* info);
extern void dbdsqr(const char* uplo, const int n, const int ncvt,
                   const int nru, const int ncc, f64* D, f64* E,
                   f64* VT, const int ldvt, f64* U, const int ldu,
                   f64* C, const int ldc, f64* work, int* info);

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
f64 dqrt12(const int m, const int n, const f64* A, const int lda,
              const f64* S, f64* work, const int lwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    int i, j, mn, info, iscl;
    f64 anrm, bignum, smlnum, nrmsvl;
    f64 dummy[1];

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
    f64 result = cblas_dasum(mn, &work[m * n], 1) /
                    (dlamch("E") * (f64)((m > n) ? m : n));
    if (nrmsvl != ZERO) {
        result /= nrmsvl;
    }

    return result;
}
