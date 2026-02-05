/**
 * @file dsbevd_2stage.c
 * @brief DSBEVD_2STAGE computes all eigenvalues and optionally eigenvectors of a
 *        real symmetric band matrix using 2-stage reduction and divide-and-conquer.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>
#include <math.h>

/**
 * DSBEVD_2STAGE computes all the eigenvalues and, optionally, eigenvectors of
 * a real symmetric band matrix A using the 2-stage technique for
 * the reduction to tridiagonal. If eigenvectors are desired, it uses
 * a divide and conquer algorithm.
 *
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     uplo    = 'U': upper triangle stored; = 'L': lower triangle stored
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      Number of super/sub-diagonals. kd >= 0.
 * @param[in,out] AB      Band matrix, overwritten on exit.
 * @param[in]     ldab    Leading dimension of AB. ldab >= kd+1.
 * @param[out]    W       Eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work    Workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    iwork   Integer workspace. On exit, iwork[0] = optimal LIWORK.
 * @param[in]     liwork  Length of iwork. If -1, workspace query.
 * @param[out]    info    = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void dsbevd_2stage(
    const char* jobz,
    const char* uplo,
    const int n,
    const int kd,
    double* restrict AB,
    const int ldab,
    double* restrict W,
    double* restrict Z,
    const int ldz,
    double* restrict work,
    const int lwork,
    int* restrict iwork,
    const int liwork,
    int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int lower, lquery, wantz;
    int iinfo, inde, indwk2, indwrk, iscale, liwmin;
    int llwork, lwmin, lhtrd = 0, lwtrd, ib, indhous, llwrk2;
    double anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (n <= 1) {
        liwmin = 1;
        lwmin = 1;
    } else {
        ib = ilaenv2stage(2, "DSYTRD_SB2ST", jobz, n, kd, -1, -1);
        lhtrd = ilaenv2stage(3, "DSYTRD_SB2ST", jobz, n, kd, ib, -1);
        lwtrd = ilaenv2stage(4, "DSYTRD_SB2ST", jobz, n, kd, ib, -1);
        if (wantz) {
            liwmin = 3 + 5 * n;
            lwmin = 1 + 5 * n + 2 * n * n;
        } else {
            liwmin = 1;
            lwmin = (2 * n > n + lhtrd + lwtrd) ? 2 * n : n + lhtrd + lwtrd;
        }
    }
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (kd < 0) {
        *info = -4;
    } else if (ldab < kd + 1) {
        *info = -6;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -9;
    }

    if (*info == 0) {
        work[0] = (double)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -11;
        } else if (liwork < liwmin && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("DSBEVD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = AB[0];
        if (wantz) {
            Z[0] = ONE;
        }
        return;
    }

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    anrm = dlansb("M", uplo, n, kd, AB, ldab, work);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            dlascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            dlascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    inde = 0;
    indhous = inde + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;
    indwk2 = indwrk + n * n;
    llwrk2 = lwork - indwk2;

    dsytrd_sb2st("N", jobz, uplo, n, kd, AB, ldab, W,
                 &work[inde], &work[indhous], lhtrd,
                 &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        dsterf(n, W, &work[inde], info);
    } else {
        dstedc("I", n, W, &work[inde], &work[indwrk], n,
               &work[indwk2], llwrk2, iwork, liwork, info);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, ONE, Z, ldz, &work[indwrk], n,
                    ZERO, &work[indwk2], n);
        dlacpy("A", n, n, &work[indwk2], n, Z, ldz);
    }

    if (iscale == 1) {
        cblas_dscal(n, ONE / sigma, W, 1);
    }

    work[0] = (double)lwmin;
    iwork[0] = liwmin;
}
