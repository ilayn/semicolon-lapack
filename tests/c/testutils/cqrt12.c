/**
 * @file cqrt12.c
 * @brief CQRT12 computes || svd(R) - s || / (||s|| * eps * max(M,N)).
 *
 * Port of LAPACK TESTING/LIN/cqrt12.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CQRT12 computes the singular values of the upper trapezoid
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
 * @param[out] work  Array (lwork). Complex workspace.
 * @param[in]  lwork The length of the array work. LWORK >= M*N + 2*min(M,N) +
 *                   max(M,N).
 * @param[out] rwork Real workspace array, dimension (2*min(M,N)).
 *
 * @return The test ratio || svd(R) - s || / (||s|| * eps * max(M,N)).
 */
f32 cqrt12(const INT m, const INT n, const c64* A, const INT lda,
              const f32* S, c64* work, const INT lwork,
              f32* rwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    INT i, j, mn, info, iscl;
    f32 anrm, bignum, smlnum, nrmsvl;
    f32 dummy[1];

    /* Quick return if possible */
    mn = (m < n) ? m : n;
    if (mn <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    if (lwork < m * n + 2 * mn + ((m > n) ? m : n)) {
        xerbla("CQRT12", 7);
        return ZERO;
    }

    /* Compute ||S|| */
    nrmsvl = cblas_snrm2(mn, S, 1);

    /* Copy upper triangle of A into work */
    claset("F", m, n, CZERO, CZERO, work, m);
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
    anrm = clange("M", m, n, work, m, dummy);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        clascl("G", 0, 0, anrm, smlnum, m, n, work, m, &info);
        iscl = 1;
    } else if (anrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        clascl("G", 0, 0, anrm, bignum, m, n, work, m, &info);
        iscl = 1;
    }

    if (anrm != ZERO) {
        /* Compute SVD of work
         * rwork(0:)        = D (diagonal, real)
         * rwork(mn:)       = E (superdiagonal, real)
         * work(m*n:)       = tauq (complex)
         * work(m*n+mn:)    = taup (complex)
         * work(m*n+2*mn:)  = workspace for cgebd2 (complex)
         */
        cgebd2(m, n, work, m,
               &rwork[0], &rwork[mn],
               &work[m * n], &work[m * n + mn],
               &work[m * n + 2 * mn], &info);

        /* Compute singular values from bidiagonal form */
        sbdsqr("U", mn, 0, 0, 0,
               &rwork[0], &rwork[mn],
               NULL, mn, NULL, 1, NULL, mn,
               &rwork[2 * mn], &info);

        if (iscl == 1) {
            if (anrm > bignum) {
                slascl("G", 0, 0, bignum, anrm, mn, 1, &rwork[0], mn, &info);
            }
            if (anrm < smlnum) {
                slascl("G", 0, 0, smlnum, anrm, mn, 1, &rwork[0], mn, &info);
            }
        }
    } else {
        for (i = 0; i < mn; i++) {
            rwork[i] = ZERO;
        }
    }

    /* Compare s and singular values of work: rwork(0:mn) -= S */
    cblas_saxpy(mn, -ONE, S, 1, &rwork[0], 1);

    /* Return || diff || / (eps * max(M,N)) / ||S|| */
    f32 result = cblas_sasum(mn, &rwork[0], 1) /
                    (slamch("E") * (f32)((m > n) ? m : n));
    if (nrmsvl != ZERO) {
        result /= nrmsvl;
    }

    return result;
}
