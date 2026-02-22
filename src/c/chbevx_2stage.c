/**
 * @file chbevx_2stage.c
 * @brief CHBEVX_2STAGE computes selected eigenvalues and optionally eigenvectors
 *        of a complex Hermitian band matrix using 2-stage reduction.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHBEVX_2STAGE computes selected eigenvalues and, optionally, eigenvectors
 * of a complex Hermitian band matrix A using the 2stage technique for
 * the reduction to tridiagonal. Eigenvalues and eigenvectors can
 * be selected by specifying either a range of values or a range of
 * indices for the desired eigenvalues.
 *
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     range   = 'A': all eigenvalues; = 'V': eigenvalues in (vl,vu];
 *                          = 'I': il-th through iu-th eigenvalues.
 * @param[in]     uplo    = 'U': upper triangle stored; = 'L': lower triangle stored
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      Number of super/sub-diagonals. kd >= 0.
 * @param[in,out] AB      Complex band matrix, overwritten on exit.
 * @param[in]     ldab    Leading dimension of AB. ldab >= kd+1.
 * @param[out]    Q       Unitary matrix if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldq     Leading dimension of Q. ldq >= 1, or ldq >= n if jobz='V'.
 * @param[in]     vl      Lower bound if range='V'.
 * @param[in]     vu      Upper bound if range='V'. vl < vu.
 * @param[in]     il      Smallest eigenvalue index if range='I'.
 * @param[in]     iu      Largest eigenvalue index if range='I'.
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       Number of eigenvalues found.
 * @param[out]    W       Selected eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work    Complex workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    rwork   Single precision workspace, dimension (7*n).
 * @param[out]    iwork   Integer workspace, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void chbevx_2stage(
    const char* jobz,
    const char* range,
    const char* uplo,
    const INT n,
    const INT kd,
    c64* restrict AB,
    const INT ldab,
    c64* restrict Q,
    const INT ldq,
    const f32 vl,
    const f32 vu,
    const INT il,
    const INT iu,
    const f32 abstol,
    INT* m,
    f32* restrict W,
    c64* restrict Z,
    const INT ldz,
    c64* restrict work,
    const INT lwork,
    f32* restrict rwork,
    INT* restrict iwork,
    INT* restrict ifail,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT alleig, indeig, lower, test, valeig, wantz, lquery;
    char order;
    INT i, iinfo, imax, indd, inde, indee, indibl;
    INT indisp, indiwk, indrwk, indwrk, iscale, itmp1, j, jj;
    INT llwork, lwmin, lhtrd = 0, lwtrd, ib, indhous, nsplit;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f32 sigma, smlnum, tmp1, vll, vuu;
    c64 ctmp1;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1);

    *info = 0;
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!alleig && !valeig && !indeig) {
        *info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (kd < 0) {
        *info = -5;
    } else if (ldab < kd + 1) {
        *info = -7;
    } else if (wantz && ldq < ((1 > n) ? 1 : n)) {
        *info = -9;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -11;
        } else if (indeig) {
            if (il < 0 || il > ((0 > n - 1) ? 0 : n - 1)) {
                *info = -12;
            } else if (iu < ((n - 1 < il) ? n - 1 : il) || iu > n - 1) {
                *info = -13;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n))
            *info = -18;
    }

    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
            work[0] = CMPLXF((f32)lwmin, 0.0f);
        } else {
            ib = ilaenv2stage(2, "CHETRD_HB2ST", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "CHETRD_HB2ST", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "CHETRD_HB2ST", jobz, n, kd, ib, -1);
            lwmin = lhtrd + lwtrd;
            work[0] = CMPLXF((f32)lwmin, 0.0f);
        }

        if (lwork < lwmin && !lquery)
            *info = -20;
    }

    if (*info != 0) {
        xerbla("CHBEVX_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0)
        return;

    if (n == 1) {
        *m = 1;
        if (lower) {
            ctmp1 = AB[0];
        } else {
            ctmp1 = AB[kd];
        }
        tmp1 = crealf(ctmp1);
        if (valeig) {
            if (!(vl < tmp1 && vu >= tmp1))
                *m = 0;
        }
        if (*m == 1) {
            W[0] = crealf(ctmp1);
            if (wantz)
                Z[0] = CONE;
        }
        return;
    }

    safmin = slamch("S");
    eps = slamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = (sqrtf(bignum) < ONE / sqrtf(sqrtf(safmin))) ? sqrtf(bignum) : ONE / sqrtf(sqrtf(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
    }
    anrm = clanhb("M", uplo, n, kd, AB, ldab, rwork);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            clascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            clascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
        if (abstol > 0)
            abstll = abstol * sigma;
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    indd = 0;
    inde = indd + n;
    indrwk = inde + n;

    indhous = 0;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    chetrd_hb2st("N", jobz, uplo, n, kd, AB, ldab,
                 &rwork[indd], &rwork[inde], &work[indhous], lhtrd,
                 &work[indwrk], llwork, &iinfo);

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_scopy(n, &rwork[indd], 1, W, 1);
        indee = indrwk + 2 * n;
        if (!wantz) {
            cblas_scopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            ssterf(n, W, &rwork[indee], info);
        } else {
            clacpy("A", n, n, Q, ldq, Z, ldz);
            cblas_scopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            csteqr(jobz, n, W, &rwork[indee], Z, ldz, &rwork[indrwk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L30;
        }
        *info = 0;
    }

    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }
    indibl = 0;
    indisp = indibl + n;
    indiwk = indisp + n;
    sstebz(range, &order, n, vll, vuu, il, iu, abstll,
           &rwork[indd], &rwork[inde], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwk], info);

    if (wantz) {
        cstein(n, &rwork[indd], &rwork[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwk], ifail, info);

        for (j = 0; j < *m; j++) {
            cblas_ccopy(n, &Z[j * ldz], 1, work, 1);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, Q, ldq,
                        work, 1, &CZERO, &Z[j * ldz], 1);
        }
    }

L30:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    if (wantz) {
        for (j = 0; j < *m - 1; j++) {
            i = 0;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i != 0) {
                itmp1 = iwork[indibl + i];
                W[i] = W[j];
                iwork[indibl + i] = iwork[indibl + j];
                W[j] = tmp1;
                iwork[indibl + j] = itmp1;
                cblas_cswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
