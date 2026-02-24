/**
 * @file sqrt12.c
 * @brief SQRT12 computes || svd(R) - s || / (||s|| * eps * max(M,N)).
 *
 * Port of LAPACK TESTING/LIN/sqrt12.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * SQRT12 computes the singular values of the upper trapezoid
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
f32 sqrt12(const INT m, const INT n, const f32* A, const INT lda,
              const f32* S, f32* work, const INT lwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    INT i, j, mn, info, iscl;
    f32 anrm, bignum, smlnum, nrmsvl;
    f32 dummy[1];

    /* Quick return if possible */
    mn = (m < n) ? m : n;
    if (mn <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    INT lwork_min1 = m * n + 4 * mn + ((m > n) ? m : n);
    INT lwork_min2 = m * n + 2 * mn + 4 * n;
    INT lwork_min = (lwork_min1 > lwork_min2) ? lwork_min1 : lwork_min2;
    if (lwork < lwork_min) {
        return ZERO;
    }

    /* Compute ||S|| */
    nrmsvl = cblas_snrm2(mn, S, 1);

    /* Copy upper triangle of A into work */
    slaset("F", m, n, ZERO, ZERO, work, m);
    for (j = 0; j < n; j++) {
        INT imax = (j + 1 < m) ? (j + 1) : m;
        for (i = 0; i < imax; i++) {
            work[j * m + i] = A[j * lda + i];
        }
    }

    /* Get machine parameters */
    smlnum = slamch("S") / slamch("P");
    bignum = ONE / smlnum;

    /* Scale work if max entry outside range [SMLNUM, BIGNUM] */
    anrm = slange("M", m, n, work, m, dummy);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        slascl("G", 0, 0, anrm, smlnum, m, n, work, m, &info);
        iscl = 1;
    } else if (anrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        slascl("G", 0, 0, anrm, bignum, m, n, work, m, &info);
        iscl = 1;
    }

    if (anrm != ZERO) {
        /* Compute SVD of work */
        /* work(m*n:)     = D (diagonal)
         * work(m*n+mn:)  = E (superdiagonal)
         * work(m*n+2*mn:) = tauq
         * work(m*n+3*mn:) = taup
         * work(m*n+4*mn:) = workspace for sgebd2 */
        sgebd2(m, n, work, m,
               &work[m * n], &work[m * n + mn],
               &work[m * n + 2 * mn], &work[m * n + 3 * mn],
               &work[m * n + 4 * mn], &info);

        /* Compute singular values from bidiagonal form */
        sbdsqr("U", mn, 0, 0, 0,
               &work[m * n], &work[m * n + mn],
               NULL, mn, NULL, 1, NULL, mn,
               &work[m * n + 2 * mn], &info);

        if (iscl == 1) {
            if (anrm > bignum) {
                slascl("G", 0, 0, bignum, anrm, mn, 1, &work[m * n], mn, &info);
            }
            if (anrm < smlnum) {
                slascl("G", 0, 0, smlnum, anrm, mn, 1, &work[m * n], mn, &info);
            }
        }
    } else {
        for (i = 0; i < mn; i++) {
            work[m * n + i] = ZERO;
        }
    }

    /* Compare s and singular values of work: work(m*n:m*n+mn) -= S */
    cblas_saxpy(mn, -ONE, S, 1, &work[m * n], 1);

    /* Return || diff || / (eps * max(M,N)) / ||S|| */
    f32 result = cblas_sasum(mn, &work[m * n], 1) /
                    (slamch("E") * (f32)((m > n) ? m : n));
    if (nrmsvl != ZERO) {
        result /= nrmsvl;
    }

    return result;
}
