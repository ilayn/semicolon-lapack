/**
 * @file zgeqp3rk.c
 * @brief ZGEQP3RK computes a truncated Householder QR factorization with column pivoting of a complex m-by-n matrix using Level 3 BLAS.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZGEQP3RK computes a truncated (rank K) or full rank Householder QR
 * factorization with column pivoting of a complex M-by-N matrix A using
 * Level 3 BLAS. K is the number of columns that were factorized.
 *
 *   A * P(K) = Q(K) * R(K)
 *
 * At the same time, the routine overwrites a complex M-by-NRHS matrix B
 * with Q(K)**H * B using Level 3 BLAS.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in] kmax
 *          The maximum number of columns to factorize. kmax >= 0.
 *
 * @param[in] abstol
 *          The absolute tolerance for maximum column 2-norm.
 *
 * @param[in] reltol
 *          The relative tolerance for maximum column 2-norm.
 *
 * @param[in,out] A
 *          Complex*16 array, dimension (lda, n+nrhs).
 *          On entry, the M-by-N matrix A and M-by-NRHS matrix B.
 *          On exit, the factors of A and Q(K)**H * B.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] K
 *          The factorization rank.
 *
 * @param[out] maxc2nrmk
 *          The maximum column 2-norm of the residual matrix.
 *
 * @param[out] relmaxc2nrmk
 *          The ratio maxc2nrmk / maxc2nrm.
 *
 * @param[out] jpiv
 *          Integer array, dimension (n). Column pivot indices.
 *
 * @param[out] tau
 *          Complex*16 array, dimension (min(m, n)).
 *
 * @param[out] work
 *          Complex*16 workspace of size (max(1, lwork)).
 *          On exit, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          lwork >= 1 if min(m,n) = 0, otherwise lwork >= n+nrhs-1.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] rwork
 *          Double precision array, dimension (2*n).
 *          Used to store partial and exact column 2-norms.
 *
 * @param[out] iwork
 *          Integer array, dimension (n-1).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = j (1 <= j <= n): NaN detected in column j
 *                         - = j (n+1 <= j <= 2*n): Inf detected in column j-n
 */
void zgeqp3rk(
    const INT m,
    const INT n,
    const INT nrhs,
    const INT kmax,
    f64 abstol,
    f64 reltol,
    c128* restrict A,
    const INT lda,
    INT* K,
    f64* maxc2nrmk,
    f64* relmaxc2nrmk,
    INT* restrict jpiv,
    c128* restrict tau,
    c128* restrict work,
    const INT lwork,
    f64* restrict rwork,
    INT* restrict iwork,
    INT* info)
{
    INT lquery, done;
    INT iinfo, ioffset, iws, j, jb, jbf, jmaxb, jmax, jmaxc2nrm;
    INT kp1, lwkopt, minmn, n_sub, nb, nbmin, nx, kf;
    f64 eps, hugeval, maxc2nrm, safmin;

    *info = 0;
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (kmax < 0) {
        *info = -4;
    } else if (disnan(abstol)) {
        *info = -5;
    } else if (disnan(reltol)) {
        *info = -6;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -8;
    }

    if (*info == 0) {
        minmn = (m < n) ? m : n;
        if (minmn == 0) {
            iws = 1;
            lwkopt = 1;
        } else {
            iws = n + nrhs - 1;
            nb = 32;
            lwkopt = 2 * n + nb * (n + nrhs + 1);
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);

        if (lwork < iws && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("ZGEQP3RK", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (minmn == 0) {
        *K = 0;
        *maxc2nrmk = 0.0;
        *relmaxc2nrmk = 0.0;
        work[0] = CMPLX((f64)lwkopt, 0.0);
        return;
    }

    for (j = 0; j < n; j++) {
        jpiv[j] = j;
    }

    for (j = 0; j < n; j++) {
        rwork[j] = cblas_dznrm2(m, &A[0 + j * lda], 1);
        rwork[n + j] = rwork[j];
    }

    kp1 = cblas_idamax(n, &rwork[0], 1);
    maxc2nrm = rwork[kp1];

    if (disnan(maxc2nrm)) {

        *K = 0;
        *info = kp1 + 1;

        *maxc2nrmk = maxc2nrm;
        *relmaxc2nrmk = maxc2nrm;

        work[0] = CMPLX((f64)lwkopt, 0.0);
        return;
    }

    if (maxc2nrm == 0.0) {

        *K = 0;
        *maxc2nrmk = 0.0;
        *relmaxc2nrmk = 0.0;

        for (j = 0; j < minmn; j++) {
            tau[j] = CMPLX(0.0, 0.0);
        }

        work[0] = CMPLX((f64)lwkopt, 0.0);
        return;

    }

    hugeval = dlamch("O");

    if (maxc2nrm > hugeval) {

        *info = n + kp1 + 1;

    }

    if (kmax == 0) {
        *K = 0;
        *maxc2nrmk = maxc2nrm;
        *relmaxc2nrmk = 1.0;
        for (j = 0; j < minmn; j++) {
            tau[j] = CMPLX(0.0, 0.0);
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);
        return;
    }

    eps = dlamch("E");

    if (abstol >= 0.0) {
        safmin = dlamch("S");
        abstol = (abstol > 2.0 * safmin) ? abstol : (2.0 * safmin);
    }

    if (reltol >= 0.0) {
        reltol = (reltol > eps) ? reltol : eps;
    }

    jmax = (kmax < minmn) ? kmax : minmn;

    if (maxc2nrm <= abstol || 1.0 <= reltol) {

        *K = 0;
        *maxc2nrmk = maxc2nrm;
        *relmaxc2nrmk = 1.0;

        for (j = 0; j < minmn; j++) {
            tau[j] = CMPLX(0.0, 0.0);
        }

        work[0] = CMPLX((f64)lwkopt, 0.0);
        return;
    }

    nbmin = 2;
    nx = 0;

    if (nb > 1 && nb < minmn) {

        nx = 128;

        if (nx < minmn) {

            if (lwork < lwkopt) {

                nb = (lwork - 2 * n) / (n + 1);
                nbmin = 2;

            }
        }
    }

    done = 0;

    j = 1;

    jmaxb = (kmax < minmn - nx) ? kmax : (minmn - nx);

    if (nb >= nbmin && nb < jmax && jmaxb > 0) {

        while (j <= jmaxb) {

            jb = (nb < jmaxb - j + 1) ? nb : (jmaxb - j + 1);
            n_sub = n - j + 1;
            ioffset = j - 1;

            zlaqp3rk(m, n_sub, nrhs, ioffset, &jb, abstol,
                     reltol, kp1, maxc2nrm, &A[0 + (j - 1) * lda], lda,
                     &done, &jbf, maxc2nrmk, relmaxc2nrmk,
                     &jpiv[j - 1], &tau[j - 1],
                     &rwork[j - 1], &rwork[n + j - 1],
                     &work[0], &work[jb],
                     n + nrhs - j + 1, iwork, &iinfo);

            if (iinfo > n_sub && *info == 0) {
                *info = 2 * ioffset + iinfo;
            }

            if (done) {

                *K = ioffset + jbf;

                if (iinfo <= n_sub && iinfo > 0) {
                    *info = ioffset + iinfo;
                }

                work[0] = CMPLX((f64)lwkopt, 0.0);

                return;

            }

            j = j + jbf;

        }

    }

    if (j <= jmax) {

        n_sub = n - j + 1;
        ioffset = j - 1;

        zlaqp2rk(m, n_sub, nrhs, ioffset, jmax - j + 1,
                 abstol, reltol, kp1, maxc2nrm, &A[0 + (j - 1) * lda], lda,
                 &kf, maxc2nrmk, relmaxc2nrmk, &jpiv[j - 1],
                 &tau[j - 1], &rwork[j - 1], &rwork[n + j - 1],
                 &work[0], &iinfo);

        *K = j - 1 + kf;

        if (iinfo > n_sub && *info == 0) {
            *info = 2 * ioffset + iinfo;
        } else if (iinfo <= n_sub && iinfo > 0) {
            *info = ioffset + iinfo;
        }

    } else {

        *K = jmax;

        if (*K < minmn) {
            jmaxc2nrm = *K + cblas_idamax(n - *K, &rwork[*K], 1);
            *maxc2nrmk = rwork[jmaxc2nrm];
            if (*K == 0) {
                *relmaxc2nrmk = 1.0;
            } else {
                *relmaxc2nrmk = *maxc2nrmk / maxc2nrm;
            }

            for (j = *K; j < minmn; j++) {
                tau[j] = CMPLX(0.0, 0.0);
            }

        }

    }

    work[0] = CMPLX((f64)lwkopt, 0.0);
}
